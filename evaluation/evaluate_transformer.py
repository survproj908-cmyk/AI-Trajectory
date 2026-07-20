import numpy as np
import torch

from models.transformer_model import TransformerPredictor

X = np.load(
    "data/trajectories/X_test.npy"
)

Y = np.load(
    "data/trajectories/Y_test.npy"
)

X = torch.tensor(
    X,
    dtype=torch.float32
)

Y = torch.tensor(
    Y,
    dtype=torch.float32
)

model = TransformerPredictor()

model.load_state_dict(
    torch.load(
        "models/transformer_model.pth"
    )
)

model.eval()

with torch.no_grad():

    pred = model(X)

pred = pred.numpy()
gt = Y.numpy()

errors = np.linalg.norm(
    pred - gt,
    axis=2
)

ADE = errors.mean()
FDE = errors[:, -1].mean()

print("ADE:", ADE)
print("FDE:", FDE)