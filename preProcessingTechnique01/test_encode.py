import numpy as np

honeynet_data = np.load("processed_datasets/honeynet_encoded.npy")
myspace_data = np.load("processed_datasets/myspace_encoded.npy")
example_data = np.load("processed_datasets/example_encoded.npy")

print("Honeynet encoded shape:", honeynet_data.shape)
print("Myspace encoded shape:", myspace_data.shape)
print("Example encoded shape: ", example_data.shape)