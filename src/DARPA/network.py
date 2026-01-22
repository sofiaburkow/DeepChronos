"""
LSTM model definitions for flow-based classification.
"""

import torch
import torch.nn as nn


class FlowLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, with_softmax=True):
        super().__init__()

        assert output_dim >= 1, "Output dim must be at least 1"

        self.with_softmax = with_softmax
        if with_softmax: # should be False while pretraining
            if output_dim == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(dim=1)

        self.lstm = nn.LSTM(
            input_size=input_dim, # MUST be n_features
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]              # (batch, hidden_dim)
        out = self.classifier(h_last) # (batch, output_dim)
        if self.with_softmax:
            out = self.softmax(out)

        return out


class EnsembleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=6):
        super().__init__()

        # create one LSTM per output class
        self.lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    batch_first=True,
                )
                for _ in range(output_dim)
            ]
        )

        # per-class head that maps the last hidden state to a scalar logit
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
                for _ in range(output_dim)
            ]
        )

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        logits = []
        for lstm, head in zip(self.lstms, self.heads):
            _, (h_n, _) = lstm(x)
            h_last = h_n[-1]  # (batch, hidden_dim)
            logit = head(h_last)  # (batch, 1)
            logits.append(logit)

        # stack per-class logits -> (batch, output_dim)
        out = torch.cat(logits, dim=1)

        return out