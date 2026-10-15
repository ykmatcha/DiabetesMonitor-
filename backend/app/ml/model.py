"""
PyTorch LSTM for multi-horizon glucose forecasting.

Input:  a sequence of recent feature vectors (e.g. last 12 readings = 60 min at 5min spacing)
Output: predicted glucose at each requested horizon (30 / 60 / 120 min ahead), predicted jointly.
"""
import torch
import torch.nn as nn


class GlucoseLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2,
                 horizons=(30, 60, 120), dropout: float = 0.2):
        super().__init__()
        self.horizons = horizons
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, len(horizons)),  # one output per horizon
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.head(last_hidden)  # (batch, n_horizons)
