import torch
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from loader.ft_orientation import make_orientation_loader
from trainer import make_FACE_trainer
from utils import set_seed


def get_subjects(data_dir):
    return [s for s in sorted(os.listdir(data_dir))
            if os.path.isdir(os.path.join(data_dir, s))]


def finetune_from_base(subject, base_path, args):
    """Fine-tune `subject` from a single pretrained base.

    Returns (best_accuracy, checkpoint_dict) where checkpoint_dict is the best
    model (or the last model if the trainer never improved on it).
    """
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    # reset RNG before each base so the runs are fair and comparable
    set_seed(42)

    # load pretrained model
    ckpt = torch.load(base_path, map_location='cpu', weights_only=False)
    model = ckpt['model']
    print(f"\n  [base={base_name}] loaded pretrained model: {ckpt['name']}")

    # create data loaders
    train_loader, test_loader = make_orientation_loader(
        data_dir=args.data_dir,
        subject=subject,
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    log_dir = os.path.join(args.log_dir, f"orientation-{subject}-{base_name}")

    f_labels = train_loader.dataset.labels['fatigue']
    f_min, f_max = float(f_labels.min()), float(f_labels.max())
    fatigue_median = float(f_labels.median())
    # fall back to midpoint when median == max (all values on one side → 0 positives)
    if fatigue_median >= f_max and f_min < f_max:
        fatigue_median = (f_min + f_max) / 2
    print(f"  [base={base_name}] Fatigue label range: {f_min}-{f_max}, threshold: {fatigue_median}"
          f" -> pos/neg: {int((f_labels > fatigue_median).sum())}/{int((f_labels <= fatigue_median).sum())}")

    trainer_config = {
        'model': model,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'criterion': criterion,
        'log_dir': log_dir,
        'target': 'fatigue',
        'epochs': args.epochs,
        'fatigue_threshold': fatigue_median,
        'select_metric': 'f1',
    }

    trainer = make_FACE_trainer(**trainer_config)
    trainer.train()

    # the trainer saves its best model to log_dir/RVT/model.pth
    trainer_model_path = os.path.join(log_dir, 'RVT', 'model.pth')
    if os.path.exists(trainer_model_path):
        best_ckpt = torch.load(trainer_model_path, map_location='cpu', weights_only=False)
    else:
        # trainer never improved — fall back to the last model
        best_ckpt = {'model': model, 'name': model.__class__.__name__}

    print(f"  [base={base_name}] best eval f1: {trainer.best_score:.4f} (acc={trainer.best_accuracy:.4f})")
    return trainer.best_score, best_ckpt


def finetune_subject(subject, args):
    print(f"\n{'='*50}")
    print(f"Fine-tuning for subject: {subject}")
    print(f"{'='*50}")

    # fine-tune from every candidate base, keep the best-scoring one
    results = []
    for base_path in args.pretrained_models:
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        f1, ckpt = finetune_from_base(subject, base_path, args)
        results.append((f1, base_name, ckpt))

    best_f1, best_base, best_ckpt = max(results, key=lambda r: r[0])
    output_path = os.path.join(args.output_dir, f"model_{subject}.pth")
    torch.save(best_ckpt, output_path)

    scores = ", ".join(f"{name}={f1:.4f}" for f1, name, _ in results)
    print(f"\n[{subject}] f1 scores: {scores}")
    print(f"[{subject}] selected base={best_base} (f1={best_f1:.4f}); saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune fatigue model per subject on orientation data")
    parser.add_argument('--pretrained_models', nargs='+',
                        default=['pre_model/model_1.pth', 'pre_model/model_9.pth'],
                        help='Candidate base models; each subject is fine-tuned from all of '
                             'them and the best-scoring result is kept')
    parser.add_argument('--data_dir', type=str, default='Orientation')
    parser.add_argument('--output_dir', type=str, default='pre_model')
    parser.add_argument('--log_dir', type=str, default='results_orientation')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--subjects', nargs='*', default=None,
                        help='Subjects to fine-tune (default: all in data_dir)')
    args = parser.parse_args()

    set_seed(42)

    subjects = args.subjects if args.subjects else get_subjects(args.data_dir)
    print(f"Subjects to fine-tune: {subjects}")

    os.makedirs(args.output_dir, exist_ok=True)

    for subject in subjects:
        finetune_subject(subject, args)

    print("\nDone. Fine-tuned models:")
    for subject in subjects:
        path = os.path.join(args.output_dir, f"model_{subject}.pth")
        exists = os.path.exists(path)
        print(f"  {path} — {'OK' if exists else 'MISSING'}")


if __name__ == '__main__':
    main()
