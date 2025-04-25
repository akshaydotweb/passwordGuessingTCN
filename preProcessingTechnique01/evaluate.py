# evaluate.py - Test the trained model
import numpy as np
import tensorflow as tf

# Load test data
print("Loading test data...")
try:
    X_test = np.load("processed_datasets/X_test.npy")
    y_test = np.load("processed_datasets/y_test.npy")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
except FileNotFoundError:
    print("Error: Test data files (X_test.npy, y_test.npy) not found.")
    print("Please ensure train.py has run successfully and created the test split.")
    exit(1)

# Load trained model (using .keras format)
print("Loading trained model...")
try:
    model = tf.keras.models.load_model('models/tcn_model_best.keras') # Correct format
    model.summary()
except Exception as e:
    print(f"Error loading model 'models/tcn_model_best.keras': {e}")
    exit(1)

# Reshape y_test to match the flattened output of the model if needed
try:
    y_test_flat = y_test.reshape(y_test.shape[0], -1)
    print(f"y_test_flat shape: {y_test_flat.shape}")
except Exception as e:
    print(f"Error reshaping y_test: {e}")
    exit(1)

# Evaluate performance using the model's compiled loss and metrics
print("Evaluating model performance on the test set...")
results = model.evaluate(X_test, y_test_flat) # Use flattened y_test
test_loss = results[0]
test_mae = results[1] if len(results) > 1 else float('nan')

print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test Mean Absolute Error (MAE): {test_mae:.4f}")

print("Evaluation script finished.")


