import os
import torch
from utils import move_batch_to_device, calculate_metrics
from accelerate import Accelerator
import numpy as np

class FACE_Trainer():
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.train_loader = kwargs['train_loader']
        self.test_loader = kwargs['test_loader']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.epochs = kwargs['epochs']
        self.log_dir = kwargs['log_dir']
        self.target = kwargs['target']
        self._prepare()

        self.best_accuracy = 0.0

    def _prepare(self):
        # prepare the model, optimizer, train_loader, and test_loader
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model = self.accelerator.prepare(self.model)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.model, self.optimizer, self.train_loader, self.test_loader = self.accelerator.prepare(
            self.model, 
            self.optimizer, 
            self.train_loader,
            self.test_loader
        )
        self._make_log_dir()
    
    def _make_log_dir(self):
        # get model class name
        model_name = self.model.__class__.__name__
        self.log_dir = os.path.join(
            self.log_dir, 
            model_name
        )
        os.makedirs(self.log_dir, exist_ok=True)

    def _train_epoch(self, epoch):
        running_loss = 0.0
        self.model.train()
        targets = []
        outputs = []
        for batch in self.train_loader:
            if batch is None:
                continue
            self.optimizer.zero_grad()
            if self.target == 'fatigue' or 'focus':
                loss, target, output = self._forward_model(batch, self.target, True)
            elif self.target == 'rt':
                loss, target, output = self._forward_model_rt(batch, True)
            else:
                raise ValueError('Invalid target')
            self.optimizer.step()
            running_loss += loss.item()
            targets.append(target)
            outputs.append(output)
        
        metrics_results = self._calculate_metrics(targets, outputs)

        logs = self._make_log(
            epoch = epoch,
            running_loss = running_loss,
            mode = 'train',
            **metrics_results
        )
        print(logs)

    def _eval_epoch(self, epoch):
        running_loss = 0.0
        self.model.eval()
        targets = []
        outputs = []
        with torch.no_grad():
            for batch in self.test_loader:
                if self.target == 'fatigue' or 'focus':
                    loss, target, output = self._forward_model(batch, self.target, False)
                elif self.target == 'rt':
                    loss, target, output = self._forward_model_rt(batch, False)
                else:
                    raise ValueError('Invalid target')
                running_loss += loss.item()
                targets.append(target)
                outputs.append(output)
        
        metrics_results = self._calculate_metrics(targets, outputs)

        logs = self._make_log(
            epoch = epoch,
            running_loss = running_loss,
            mode = 'eval',
            **metrics_results
        )
        print(logs)
        if metrics_results['accuracy'] > self.best_accuracy:
            self.best_accuracy = metrics_results['accuracy']
            self._save_best_results(targets, outputs)

    def _save_best_results(self, target, output):
        metrics_results = self._calculate_metrics(output, target)
        target, output = np.concatenate(target), np.concatenate(output)
        
        ckpt = {
            'target': target,
            'output': output,
        }
        # add metrics results to the checkpoint
        ckpt.update(metrics_results)
        np.save(os.path.join(self.log_dir, 'best_results.npy'), ckpt)
        self._save_model()
        print(f"Best results saved to {self.log_dir}")

    def _save_model(self):
        dict_config = {
            'model': self.model,
            'name': self.model.__class__.__name__,
        }
        torch.save(dict_config, os.path.join(self.log_dir, 'model.pth'))

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            self._eval_epoch(epoch)

    def _forward_model(self, batch, task, is_train):
        batch = move_batch_to_device(batch, self.device)
        faces = batch['seq']
        if task == 'fatigue':
            target = batch['tiredness']
        elif task == 'focus':
            target = batch['focus']
        else:
            raise ValueError('Invalid task')    

        loss = 0.0
        corr = 0
        _outputs = []
        _targets = []
        h_t = torch.zeros(2, 1, 1024).to(self.device)

        for i in range(faces.size(1)):
            block_faces = faces[:, i, :, :, :]
            block_target = target

            output, h_t = self.model(block_faces, h_t)

            results = self._compute_loss(output, block_target)
            block_loss = results['loss']
            corr += results['corr']
            loss += block_loss
            _targets.append(results['target'].cpu().numpy())
            _outputs.append(results['output'].cpu().numpy())

        if is_train:
            self.accelerator.backward(loss)
        
        return block_loss, np.concatenate(_targets), np.concatenate(_outputs)
    
    def _forward_model_rt(self, batch, is_train):
        batch = move_batch_to_device(batch, self.device)
        faces = batch['seq']
        rt = batch['rt']

        loss = torch.tensor(0.0).to(self.device)
        _outputs = []
        _targets = []

        for i in range(faces.size(1)):
            block_faces = faces[:, i, :, :, :]
            block_target = rt[:, i]

            output, h_t = self.model(block_faces, h_t)

            results = self._compute_loss(output, block_target)
            block_loss = results['loss']
            loss += block_loss.item()
            _targets.append(results['target'].cpu().numpy())
            _outputs.append(results['output'].cpu().numpy())

            if is_train:
                self.accelerator.backward(block_loss)
        
        return block_loss, np.concatenate(_targets), np.concatenate(_outputs)
    
    def _compute_loss(self, output, target):
        if self.target == 'fatigue' or 'focus':
            target = (target > 2).long()
        elif self.target == 'rt':
            target = (target > 2).long()
        else:
            raise ValueError('Invalid target')
        
        loss = self.criterion(output, target)
        corr = (output.argmax(1) == target).sum().item()
        return {
            'loss': loss,
            'target': target,
            'output': output.argmax(1),
            'corr': corr
        }
    
    def _calculate_metrics(self, outputs, targets):
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)
        return calculate_metrics(outputs, targets)

    def _make_log(self, 
                  epoch, 
                  running_loss, 
                  mode="train",
                  **kwargs):
        if mode == "train":
            num_samples = len(self.train_loader)
        elif mode in ["eval", "test"]:
            num_samples = len(self.test_loader)
        else:
            raise ValueError(f"Invalid mode '{mode}'")
        logs = {
            'epoch': epoch,
            f'{mode}_loss': running_loss/(num_samples * 2),
            f'{mode}_accuracy': kwargs['accuracy'],
            f'{mode}_f1': kwargs['f1'],
            f'{mode}_precision': kwargs['precision'],
            f'{mode}_recall': kwargs['recall']
        }
        return logs