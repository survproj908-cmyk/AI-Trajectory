import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from models.transformer_model import TransformerPredictor

X = np.load(
    "data/trajectories/X_train.npy"
)

Y = np.load(
    "data/trajectories/Y_train.npy"
)

X = torch.tensor(
    X,
    dtype=torch.float32
)

Y = torch.tensor(
    Y,
    dtype=torch.float32
)

dataset = TensorDataset(
    X,
    Y
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

model = TransformerPredictor()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 10

for epoch in range(EPOCHS):

    total_loss = 0

    for batch_x, batch_y in loader:

        pred = model(batch_x)

        loss = criterion(
            pred,
            batch_y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}: {total_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "models/transformer_model.pth"
)

print("Training Complete")