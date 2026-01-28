import torch.nn as nn

class RecurrentClassifier(nn.Module):
    def __init__(self,
                 seq_length=512,
                 num_features=1024, 
                 hidden_size=1024,
                 num_layers=2,
                 num_classes=2,
                 batch_first=True):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*seq_length, num_classes),
            nn.Dropout(0.1),
            # nn.Softmax(dim=1)
        )

    def forward(self, x, h_t):
        x, h_t = self.rnn(x, h_t)
        x = x.flatten(1)
        x = self.classifier(x)
        return x, h_t