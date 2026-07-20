import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):

    def __init__(
        self,
        input_size=2,
        hidden_size=64,
        num_layers=2,
        pred_len=4
    ):

        super().__init__()

        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            pred_len * 2
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        out = out.view(
            -1,
            self.pred_len,
            2
        )

        return out