import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def set_seed(seed):
    # set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('seed set to {}'.format(seed))

def move_batch_to_device(batch, device):
    # if batch values are tensors, move them to device
    for key in batch.keys():
        if isinstance(batch[key], dict):
            batch[key] = move_batch_to_device(batch[key], device)
        elif isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch

def get_npy_files(log_dir, target='fatigue'):
    # get all npy files under log_dir
    npy_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.npy'):
                _npy_file_path = os.path.join(root, file)
                if f"target-{target}" in _npy_file_path:
                    npy_files.append(_npy_file_path)
    
    return npy_files

def calculate_metrics(outputs, targets):
    accuracy = accuracy_score(targets, outputs)
    f1 = f1_score(targets, outputs)
    precision = precision_score(targets, outputs)
    recall = recall_score(targets, outputs)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }