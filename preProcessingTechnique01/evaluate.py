
# evaluate.py - Test the trained model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load test data
X_test = np.load("processed_datasets/X_test.npy")
y_test = np.load("processed_datasets/y_test.npy")

# Load trained model
model = tf.keras.models.load_model('models/tcn_model.h5')

# Evaluate performance
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Predict on test data
predictions = model.predict(X_test)

# Analyze character-level prediction accuracy
character_accuracy = np.mean(np.argmax(predictions, axis=2) == np.argmax(y_test, axis=2))
print(f"Character-level accuracy: {character_accuracy:.4f}")


