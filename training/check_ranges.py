import numpy as np

X = np.load(
    "data/trajectories/X.npy"
)

print("Min:", X.min())
print("Max:", X.max())