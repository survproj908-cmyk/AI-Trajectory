import numpy as np
from sklearn.model_selection import train_test_split

X = np.load(
    "data/trajectories/X_norm.npy"
)

Y = np.load(
    "data/trajectories/Y_norm.npy"
)

X_train, X_temp, Y_train, Y_temp = train_test_split(
    X,
    Y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp,
    Y_temp,
    test_size=0.50,
    random_state=42
)

np.save(
    "data/trajectories/X_train.npy",
    X_train
)

np.save(
    "data/trajectories/Y_train.npy",
    Y_train
)

np.save(
    "data/trajectories/X_val.npy",
    X_val
)

np.save(
    "data/trajectories/Y_val.npy",
    Y_val
)

np.save(
    "data/trajectories/X_test.npy",
    X_test
)

np.save(
    "data/trajectories/Y_test.npy",
    Y_test
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)