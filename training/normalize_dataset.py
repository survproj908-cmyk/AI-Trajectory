import numpy as np

X = np.load(
    "data/trajectories/X.npy"
)

Y = np.load(
    "data/trajectories/Y.npy"
)

MIN_VAL = 33
MAX_VAL = 3808

X_norm = (
    X - MIN_VAL
) / (
    MAX_VAL - MIN_VAL
)

Y_norm = (
    Y - MIN_VAL
) / (
    MAX_VAL - MIN_VAL
)

np.save(
    "data/trajectories/X_norm.npy",
    X_norm
)

np.save(
    "data/trajectories/Y_norm.npy",
    Y_norm
)

print("Normalization complete")

print(
    "X min:",
    X_norm.min()
)

print(
    "X max:",
    X_norm.max()
)