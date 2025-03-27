import numpy as np

honeynet_data = np.load("honeynet_encoded.npy")
myspace_data = np.load("myspace_encoded.npy")

print("Honeynet encoded shape:", honeynet_data.shape)
print("Myspace encoded shape:", myspace_data.shape)
