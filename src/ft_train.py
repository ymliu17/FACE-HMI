import torch
import argparse
from loader import make_FACE_loader
from utils import set_seed
from models import RVT
from trainer import make_FACE_trainer
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str, default='FACE')
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--test_subject', type=str, default='F4004')
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--log_dir', type=str, default='results_face')
argparser.add_argument('--target', type=str, default='fatigue')

args = argparser.parse_args()

set_seed(42)

train_loader, test_loader = make_FACE_loader(
    data_dir=args.data_dir, 
    batch_size=args.batch_size,
    test_subject=args.test_subject
)

model = RVT()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = torch.nn.CrossEntropyLoss()

assert args.target in ['fatigue', 'focus', 'rt'], "Invalid target"
log_dir = os.path.join(args.log_dir, 
                       f"target-{args.target}",
                       f"sub-{args.test_subject}")

trainer_config = {
    'model': model,
    'train_loader': train_loader,
    'test_loader': test_loader,
    'optimizer': optimizer,
    'criterion': criterion,
    'log_dir': log_dir,
    'target': args.target,
    'epochs': args.epochs
}

trainer = make_FACE_trainer(**trainer_config)
trainer.train()