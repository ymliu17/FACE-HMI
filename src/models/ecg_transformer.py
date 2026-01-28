import torch.nn as nn
import torch
from transformers import GPT2Model, GPT2Config
from models.recurrent_classifier import RecurrentClassifier

class ECGTransformer(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.time_embedding = TimeEmbedding(num_features)
        self.config = GPT2Config.from_pretrained(
            "gpt2", 
            n_embd=num_features,
            n_head=16,
            output_hidden_states=False
        )
        self.gpt2 = GPT2Model(self.config)

    def forward(self, 
                x, 
                attn_mask,
                time
                ):
        # Time embedding
        x = self.time_embedding(x, time)

        # GPT2
        x = self.gpt2(
            inputs_embeds=x, 
            attention_mask=attn_mask)['last_hidden_state']

        return x
    
class TimeEmbedding(nn.Module):
    def __init__(self, 
                 num_features):
        super().__init__()
        self.time_embeddings = nn.Sequential(
            nn.Linear(1, num_features),
            nn.LayerNorm(num_features),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x, time):
        # x expected shape: (batch_size, seq_length, d_model)
        # Add time embeddings to the input
        time_embeds = self.time_embeddings(time.unsqueeze(-1))
        return x + time_embeds

class RET(nn.Module):
    def __init__(self, 
                 num_features, 
                 num_classes, 
                 seq_length):
        super().__init__()
        self.ecg_transformer = ECGTransformer(
            num_features=num_features
            )

        # self.classifier = nn.Linear(num_features * seq_length, num_classes)
        self.recurrent_classifier = RecurrentClassifier(
            seq_length=seq_length,
            num_features=num_features, 
            hidden_size=num_features,
            num_layers=2,
            num_classes=num_classes,
            batch_first=True)

    def forward(self, 
                x, 
                attn_mask,
                time,
                h_t):
        # Time embedding
        x = self.ecg_transformer(x, attn_mask, time)
        x, h_t = self.recurrent_classifier(x, h_t)

        return x, h_t
    