import numpy as np
import torch

from models.lstm_model import LSTMPredictor
from evaluation.threat_detection import check_intrusion

MIN_VAL = 33
MAX_VAL = 3808

X = np.load(
    "data/trajectories/X_test.npy"
)

model = LSTMPredictor()

model.load_state_dict(
    torch.load(
        "models/lstm_model.pth"
    )
)

model.eval()

sample = torch.tensor(
    X[:10],
    dtype=torch.float32
)

with torch.no_grad():

    pred = model(sample)

pred = pred.numpy()

pred = (
    pred *
    (MAX_VAL - MIN_VAL)
) + MIN_VAL

for i in range(len(pred)):

    threat = check_intrusion(
        pred[i]
    )

    if threat:

        print(
            f"Trajectory {i}: THREAT ALERT"
        )

    else:

        print(
            f"Trajectory {i}: SAFE"
        )