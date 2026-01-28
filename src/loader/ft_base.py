import torch
import torch.utils.data as data
import pandas as pd
import os
import numpy as np
from PIL import Image
from torchvision import transforms

def get_subjects(data_dir):
        return [sub for sub in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, sub))]

class Facedataset(data.Dataset):
    def __init__(self, 
                 data_dir, 
                 subjects = None,
                 transform = None,
                 seq_len = 16):
        self.data_dir = os.path.join(data_dir, 'prep_img')
        self.labels = pd.read_csv(os.path.join(data_dir, 'vid_labels.csv'))
        self.transform = transform
        self.seq_len = seq_len

        self.subjects = subjects if subjects is not None else [sub for sub in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, sub))]
        print(f"Subjects: {self.subjects}")

        if subjects is not None:
            self.labels = self.labels[self.labels['Subject'].isin(subjects)].reset_index(drop=True)
            self.labels = self.labels[self.labels.apply(
                lambda row: os.path.exists(os.path.join(
                    self.data_dir, row['Subject'], f"ses-{row['Session']}", f"block-{row['Block']}"
                )), axis=1)].reset_index(drop=True)



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        idx = self.labels.iloc[index]['Sub_Ses_Block']
        subject = self.labels.iloc[index]['Subject']
        session = self.labels.iloc[index]['Session']
        block = self.labels.iloc[index]['Block']
        block_path = os.path.join(self.data_dir, subject, f"ses-{session}", f"block-{block}")
        faces = self._get_faces(block_path, self.seq_len)

        if faces is None:
            print(f"Warning: Block {block_path} does not exist")
            pass

        std_rt = self.labels.iloc[index]['Std_RT']
        pre_fatigue = self.labels.iloc[index]['PreFatigue']
        post_fatigue = self.labels.iloc[index]['PostFatigue']
        tiredness = self.labels.iloc[index]['Tiredness']
        focus = self.labels.iloc[index]['Focus']

        block_seq = {'idx': idx,
                     'seq': faces,
                     'rt': std_rt,
                     'pre_fatigue': pre_fatigue.astype(np.float32),
                     'post_fatigue': post_fatigue.astype(np.float32),
                     'tiredness': tiredness.astype(np.float32),
                     'focus': focus.astype(np.float32)
                     }
        
        return block_seq
    
    def _get_faces(self, block_path, seq_len):
        if os.path.exists(block_path):
            face_paths = [os.path.join(block_path, f) for f in sorted(os.listdir(block_path)) if f.lower().endswith('.jpg')]
            
            if not face_paths:
                print(f"No faces found in {block_path}")
                return None
            
            if len(face_paths) < seq_len:
                face_paths += [face_paths[-1]] * (seq_len - len(face_paths))

            seq_interval = (len(face_paths) - 1) // (seq_len - 1)
            faces = []
            for i in range(seq_len):
                if i == seq_len - 1:
                    face = Image.open(face_paths[-1])
                    face = transforms.ToTensor()(face)
                    face = transforms.Resize((112, 112), antialias=True)(face)
                    faces.append(face)
                else:
                    face = Image.open(face_paths[i * seq_interval])
                    face = transforms.ToTensor()(face)
                    face = transforms.Resize((112, 112), antialias=True)(face)
                    faces.append(face)
        
            faces = torch.stack(faces)
            return faces
        else:
            return None