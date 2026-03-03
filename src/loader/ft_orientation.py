import torch
import torch.utils.data as data
import pandas as pd
import os
import numpy as np
from PIL import Image
from torchvision import transforms


def _find_label_path(data_dir, subject):
    """Try multiple naming conventions to locate the label CSV."""
    session = f"ses-{subject}_0"
    candidates = [
        os.path.join(data_dir, subject, f"{session}_label.csv"),  # canonical
        os.path.join(data_dir, subject, f"{session}_lable.csv"),  # legacy typo
        os.path.join(data_dir, subject, f"{subject}_0.csv"),      # bare subject style
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No label CSV found for {subject}. Tried:\n" + "\n".join(f"  {p}" for p in candidates)
    )


class OrientationDataset(data.Dataset):
    def __init__(self, data_dir, subject, seq_len=16):
        self.seq_len = seq_len
        self.subject = subject

        label_path = _find_label_path(data_dir, subject)
        labels = pd.read_csv(label_path, skipinitialspace=True)
        self.block_dir = os.path.join(data_dir, subject, f"ses-{subject}_0")

        # keep only numeric block_ids (drop rows like "Pre")
        labels = labels[pd.to_numeric(labels['block_id'], errors='coerce').notna()].copy()
        labels['block_id'] = labels['block_id'].astype(int)
        # drop rows with missing fatigue
        labels = labels[labels['fatigue'].notna() & (labels['fatigue'].astype(str).str.strip() != '')].copy()
        labels['fatigue'] = labels['fatigue'].astype(int)

        # try direct match first
        direct = labels[labels['block_id'].apply(
            lambda bid: os.path.isdir(os.path.join(self.block_dir, f"block-{bid}"))
        )].reset_index(drop=True)

        if len(direct) > 0:
            self.labels = direct
        else:
            # sequential fallback: map Nth CSV row to Nth sorted block dir
            actual_blocks = sorted(
                [d for d in os.listdir(self.block_dir)
                 if d.startswith('block-') and os.path.isdir(os.path.join(self.block_dir, d))],
                key=lambda x: int(x.split('-')[1])
            )
            n = min(len(labels), len(actual_blocks))
            self.labels = labels.iloc[:n].copy().reset_index(drop=True)
            self.labels['block_id'] = [int(b.split('-')[1]) for b in actual_blocks[:n]]

        print(f"[OrientationDataset] {subject}: {len(self.labels)} blocks, "
              f"fatigue range {self.labels['fatigue'].min()}-{self.labels['fatigue'].max()} "
              f"(label: {os.path.basename(label_path)})")

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
