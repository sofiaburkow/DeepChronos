import torch
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
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]

        # print(x.shape)

        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]              # (batch, hidden_dim)

        # out = self.classifier(h_last) # (batch, 1)
        # print(out)
        # print(out.shape)
        # return out         # (batch,)

        logits = self.classifier(h_last)  # (batch, 1)
        p1 = torch.sigmoid(logits)        # (batch, 1)
        p0 = 1.0 - p1                     # (batch, 1)
        out = torch.cat([p0, p1], dim=1)  # (batch, 2)
        print(out)
        return out
    

class FlowLSTMWrapper(nn.Module):
    def __init__(self, lstm: nn.Module, classifier: nn.Module = None):
        super().__init__()
        self.lstm = lstm
        self.classifier = classifier

        # Freeze the LSTM
        for p in self.lstm.parameters():
            p.requires_grad = False

        self.lstm.eval()

    def forward(self, x):
        # x: [1, seq_len, input_dim]
        with torch.no_grad():
            out = self.lstm(x)  # frozen LSTM

        out = out.detach()       # remove LSTM gradient history
        out.requires_grad_(True) # allow gradient for downstream logic

        if self.classifier:
            out = self.classifier(out)

        print(out)
        return out