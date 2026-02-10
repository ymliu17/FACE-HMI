import torch
import torch.utils.data as data
import pandas as pd
import os
import numpy as np
from PIL import Image
from torchvision import transforms


class OrientationDataset(data.Dataset):
    def __init__(self, data_dir, subject, seq_len=16):
        self.seq_len = seq_len
        self.subject = subject

        label_path = os.path.join(data_dir, subject, f"ses-{subject}_0_lable.csv")
        self.labels = pd.read_csv(label_path, skipinitialspace=True)
        self.block_dir = os.path.join(data_dir, subject, f"ses-{subject}_0")

        # filter to blocks that actually exist on disk
        self.labels = self.labels[self.labels['block_id'].apply(
            lambda bid: os.path.isdir(os.path.join(self.block_dir, f"block-{bid}"))
        )].reset_index(drop=True)

        print(f"[OrientationDataset] {subject}: {len(self.labels)} blocks, "
              f"fatigue range {self.labels['fatigue'].min()}-{self.labels['fatigue'].max()}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.labels.iloc[index]
        block_id = row['block_id']
        fatigue = row['fatigue']
        block_path = os.path.join(self.block_dir, f"block-{block_id}")
        faces = self._get_faces(block_path, self.seq_len)

        block_seq = {
            'idx': f"{self.subject}_{block_id}",
            'seq': faces,
            'tiredness': np.float32(fatigue),
        }
        return block_seq

    def _get_faces(self, block_path, seq_len):
        face_paths = [os.path.join(block_path, f)
                      for f in sorted(os.listdir(block_path))
                      if f.lower().endswith('.jpg')]

        if not face_paths:
            raise RuntimeError(f"No faces found in {block_path}")

        if len(face_paths) < seq_len:
            face_paths += [face_paths[-1]] * (seq_len - len(face_paths))

        seq_interval = (len(face_paths) - 1) // (seq_len - 1)
        faces = []
        for i in range(seq_len):
            if i == seq_len - 1:
                face = Image.open(face_paths[-1])
            else:
                face = Image.open(face_paths[i * seq_interval])
            face = transforms.ToTensor()(face)
            face = transforms.Resize((112, 112), antialias=True)(face)
            faces.append(face)

        return torch.stack(faces)


def make_orientation_loader(data_dir, subject, batch_size=4):
    dataset = OrientationDataset(data_dir=data_dir, subject=subject)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    # use same loader for train and eval (small-data fine-tuning)
    return loader, loader
