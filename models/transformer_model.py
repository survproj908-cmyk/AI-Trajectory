import torch
import torch.nn as nn

class TransformerPredictor(nn.Module):

    def __init__(
        self,
        input_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        pred_len=4
    ):

        super().__init__()

        self.pred_len = pred_len

        self.embedding = nn.Linear(
            input_dim,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(
            d_model,
            pred_len * 2
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.transformer(x)

        x = x[:, -1, :]

        x = self.fc(x)

        x = x.view(
            -1,
            self.pred_len,
            2
        )

        return x