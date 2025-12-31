import torch.nn as nn


class FlowLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim, # MUST be n_features
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]                # (batch, hidden_dim)
        out = self.classifier(h_last)   # (batch, 1)
        return out.squeeze(1)           # (batch,)
