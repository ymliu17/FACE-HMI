# code for video ecg recurrent transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ST_Former import GenerateModel
from models.ecg_transformer import ECGTransformer
from models.recurrent_classifier import RecurrentClassifier

class VERT(nn.Module):
    def __init__(self, 
                 video_num_features=512,
                 ecg_num_features=1024,
                 video_seq_len=1,
                 ecg_seq_len=512,
                 num_classes=2,
                 proj_dim=512
                 ):
        super().__init__()
        self.generate_model = GenerateModel()
        self.ecg_transformer = ECGTransformer(
            num_features=ecg_num_features
        )
        self.recurrent_classifier = RecurrentClassifier(
            seq_length=2,
            num_features=proj_dim, 
            hidden_size=1024,
            num_classes=num_classes
        )

        self.video_proj = nn.Linear(video_num_features * video_seq_len, proj_dim)
        self.ecg_proj = nn.Linear(ecg_num_features * ecg_seq_len, proj_dim)

        self.video_param = nn.Parameter(torch.randn(1, video_num_features * video_seq_len))
        self.ecg_param = nn.Parameter(torch.randn(1, ecg_num_features * ecg_seq_len))
        
    def forward(self, video_x, ecg_x, attn_mask, time, h_t):
        # video modal
        if isinstance(video_x, list):
            proj_video_x = self.video_proj(self.video_param).unsqueeze(1)
        else:
            video_x = self.generate_model(video_x)
            video_x = video_x.flatten(1)
            proj_video_x = self.video_proj(video_x).unsqueeze(1)

        # video modal
        ecg_x = self.ecg_transformer(ecg_x, attn_mask, time)
        ecg_x = ecg_x.flatten(1)
        proj_ecg_x = self.ecg_proj(ecg_x).unsqueeze(1)

        x = torch.cat((proj_video_x, proj_ecg_x), dim=1)
        x, h_t = self.recurrent_classifier(x, h_t)
        return x, h_t