from loader.ft_base import *
import torch
import pandas as pd
import os


def make_FACE_loader(data_dir, 
                transform = None, 
                batch_size = 32, 
                shuffle = True, 
                test_subject = 'F4004',
                num_workers = 0):
    print(f"Data directory: {data_dir}, test subject: {test_subject}")
    
    test_subject = test_subject
    subjects = get_subjects(os.path.join(data_dir, 'prep_img'))
    train_subjects = [s for s in subjects if s != test_subject]
    # print(f"Train subjects: {train_subjects}")

    train_dataset = Facedataset(data_dir=data_dir, 
                                subjects=train_subjects, 
                                transform=transform, 
                                seq_len=16)
    print(f"Created train dataset with {len(train_dataset)} blocks")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    test_dataset = Facedataset(data_dir = data_dir, 
                               subjects = [test_subject,], 
                               transform = transform, 
                               seq_len=16)
    print(f"Created test dataset with {len(test_dataset)} blocks")
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return train_loader, test_loader