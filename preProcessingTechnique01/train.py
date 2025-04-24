# train.py - Implement the training pipeline
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load preprocessed data
X_train = np.load("processed_datasets/X_train.npy")
y_train = np.load("processed_datasets/y_train.npy")
X_val = np.load("processed_datasets/X_val.npy")
y_val = np.load("processed_datasets/y_val.npy")

# Define model parameters
input_shape = X_train.shape[1:]
output_shape = 4  # Based on our 4D encoding

# Build model
model = build_tcn_model(input_shape, output_shape)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('models/tcn_model.h5', save_best_only=True)
]

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=callbacks
)
