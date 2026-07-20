import json
import numpy as np

INPUT_LEN = 8
PRED_LEN = 4

with open(
    "data/trajectories/trajectories.json",
    "r"
) as f:

    trajectories = json.load(f)

X = []
Y = []

for track_id, points in trajectories.items():

    if len(points) < INPUT_LEN + PRED_LEN:
        continue

    for i in range(
        len(points) - INPUT_LEN - PRED_LEN + 1
    ):

        input_seq = points[
            i:i+INPUT_LEN
        ]

        target_seq = points[
            i+INPUT_LEN:
            i+INPUT_LEN+PRED_LEN
        ]

        X.append(input_seq)
        Y.append(target_seq)

X = np.array(X)
Y = np.array(Y)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

np.save(
    "data/trajectories/X.npy",
    X
)

np.save(
    "data/trajectories/Y.npy",
    Y
)