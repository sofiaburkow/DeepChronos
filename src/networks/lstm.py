import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim=64, 
            output_dim=2, 
            with_softmax=True
    ):
        super().__init__()
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

        return self.softmax(out) if self.with_softmax else out


class EnsembleLSTMClassifier(nn.Module):
    def __init__(
            self, 
            input_dim, 
            n_networks,      # number of attack-phase subnetworks
            output_dim,      # total number of prediction classes (including benign)
            hidden_dim=64, 
            with_softmax=True
        ):
        super().__init__()
        self.with_softmax = with_softmax
        if with_softmax: # should be False while pretraining
            if output_dim == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(dim=1)

        # create one LSTM per attack phase
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            for _ in range(n_networks)
        ])

        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim * n_networks, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        features = []
        for lstm in self.lstms:
            _, (h_n, _) = lstm(x)
            features.append(h_n[-1])  # (batch, hidden_dim)
        # concatenate outputs from each sub-LSTM
        concat = torch.cat(features, dim=1)  # (batch, hidden_dim * n_networks)
        logits = self.classifier_head(concat)  # (batch, output_dim)
        return self.softmax(logits) if self.with_softmax else logits