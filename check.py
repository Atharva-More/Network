import sys
import numpy as np
import tensorflow as tf

print("Python:", sys.version.split()[0])
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Load one dataset to confirm the data is in the right place
path = "data/UCR_TS_Archive_2015/Adiac/Adiac_TRAIN"
data = np.loadtxt(path, delimiter=",")
y, X = data[:, 0], data[:, 1:]
print("Adiac train shape:", X.shape, "| classes:", len(np.unique(y)))