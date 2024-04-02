import numpy as np

# Load the data
data = np.load('data/chair_train.npy')

# Print the type, shape, and first few elements of the data
print("Type: ", type(data))
print("Shape: ", data.shape)
print("First few elements: ", data.flatten()[:10])