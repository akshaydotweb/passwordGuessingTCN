# train.py - Implement the training pipeline
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# --- Configuration ---
PROCESSED_DIR = "processed_datasets"
MODELS_DIR = "models"
# Define which datasets to use for training/validation (adjust as needed)
# These should match the base names of the files processed by preprocess.py
DATASET_NAMES = [
    'rockyou'
    # ,
    # 'myspace',
    # '000webhost',
    # 'Ashley-Madison',
    # 'phpbb',
    # 'honeynet',
    # '10-million-passwords',
    # 'hotmail',
    # 'NordVPN',
    # 'singles.org'
    # Add other dataset names here if needed
]
SEQUENCE_LENGTH = 20 # IMPORTANT: Adjust based on your encoding/padding strategy
VALIDATION_SPLIT = 0.2 # Use 20% of data for validation
TEST_SPLIT = 0.1 # Use 10% of total data for testing
RANDOM_STATE = 42 # For reproducible train/val/test split
# --- End Configuration ---

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def create_sequences_from_flat_data(flat_data, seq_length):
    """
    Transforms a flat sequence of encoded vectors into overlapping sequences.
    Assumes 'flat_data' is a 2D numpy array [total_steps, num_features].
    """
    X, y = [], []
    print(f"Creating sequences from flat data with shape: {flat_data.shape}")
    if len(flat_data) <= seq_length:
         raise ValueError(f"Total data length ({len(flat_data)}) is not greater than sequence length ({seq_length}). Cannot create sequences.")

    # Iterate up to the point where a full sequence + target can be formed
    for i in range(len(flat_data) - seq_length):
        sequence = flat_data[i : i + seq_length]
        target = flat_data[i + seq_length] # Predict the next vector
        X.append(sequence)
        y.append(target)

    if not X:
        # This check might be redundant given the length check above, but kept for safety
        raise ValueError("No sequences generated despite sufficient data length. Check logic.")

    print(f"Generated {len(X)} sequences.")
    return np.array(X), np.array(y)

# Load and combine encoded data from specified datasets
all_encoded_data = []
print("Loading encoded datasets...")
for name in DATASET_NAMES:
    file_path = os.path.join(PROCESSED_DIR, f"{name}_encoded.npy")
    if os.path.exists(file_path):
        try:
            # Allow pickling needed for loading object arrays if your encoding uses them
            data = np.load(file_path, allow_pickle=True)
            # Handle potential nesting if data is saved incorrectly
            if data.ndim == 1 and isinstance(data[0], (np.ndarray, list)):
                 # If it looks like an array of arrays/lists, try stacking
                 try:
                     data = np.stack(data)
                 except ValueError as e:
                     print(f"Warning: Could not stack data from {file_path}. Skipping. Error: {e}")
                     continue
            elif data.ndim == 0: # Handle 0-dimensional array (scalar array)
                 print(f"Warning: Loaded 0-dimensional array from {file_path}. Skipping.")
                 continue

            if data.ndim >= 2: # Expecting at least 2D array [passwords, features]
                all_encoded_data.append(data)
                print(f"Loaded {file_path}, shape: {data.shape}")
            else:
                print(f"Warning: Loaded data from {file_path} has unexpected shape {data.shape}. Skipping.")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    else:
        print(f"Warning: File not found - {file_path}")

if not all_encoded_data:
    raise FileNotFoundError(f"No encoded .npy files found in {PROCESSED_DIR} for the specified datasets. Run preprocess.py first.")

# Concatenate all loaded data
# Ensure consistent shapes before concatenating if necessary (padding might handle this)
try:
    combined_data = np.concatenate(all_encoded_data, axis=0)
    print(f"Combined data shape: {combined_data.shape}")
except ValueError as e:
    print(f"Error concatenating datasets: {e}. Ensure all loaded datasets have compatible shapes after potential stacking.")
    # Attempt to handle varying lengths by padding (requires knowing max length and pad value)
    # This is complex and depends heavily on your encoding details.
    # As a fallback, we'll proceed only if concatenation worked.
    exit(1)

# --- TEMPORARY: Use only a subset for testing memory ---
subset_size = 500000 # Adjust this number as needed
if len(combined_data) > subset_size:
    print(f"Using a subset of {subset_size} data points for memory testing.")
    combined_data = combined_data[:subset_size]
# --- END TEMPORARY SUBSET ---

# Create sequences (X) and targets (y)
print("Creating sequences...")
# Assuming combined_data is now a 2D array [num_passwords * avg_len, features_dim1, features_dim2]
# Reshape combined_data before creating sequences if it's not already flat [total_steps, features]
# Example: if combined_data is (num_passwords, avg_len, 16, 4), reshape it first.
# Assuming combined_data is already [total_steps, 16, 4] from preprocessing
X, y = create_sequences_from_flat_data(combined_data, SEQUENCE_LENGTH)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# --- Configuration for Splitting ---
VALIDATION_SPLIT = 0.2 # Use 20% of (train+val) data for validation
TEST_SPLIT = 0.1 # Use 10% of total data for testing
RANDOM_STATE = 42 # For reproducible train/val/test split
# --- End Configuration ---

# Split data into training, validation, and test sets
print("Splitting data into training, validation, and test sets...")

# First split: Separate test set (e.g., 10% of total)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE
)

# Calculate validation split size relative to the remaining data (X_temp)
relative_val_split = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)

# Second split: Separate training and validation sets from the temporary set
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=relative_val_split, random_state=RANDOM_STATE
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- Save the test set ---
print(f"Saving test set to {PROCESSED_DIR}...")
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
print("Test set saved.")
# --- End Save Test Set ---

# Define model parameters
# Input shape should be [sequence_length, num_features_dim1, num_features_dim2]
input_shape = X_train.shape[1:] # Should be (20, 16, 4)
# Output shape should match the shape of a single target vector 'y'
output_shape_dims = y_train.shape[1:] # Should be (16, 4)
# Calculate the total number of output features (flattened)
num_output_features = np.prod(output_shape_dims) # 16 * 4 = 64

# --- IMPORTANT ---
# The model will predict a flattened vector of size 64.
# The loss function should be suitable for comparing these vectors, e.g., MSE.

# --- Build and Compile Model ---
# Make sure build_tcn_model is defined or imported elsewhere in your project
# from model import build_tcn_model # Example import

# Placeholder for build_tcn_model if not defined
def build_tcn_model(input_shape, num_output_features):
    # Replace with your actual TCN model definition
    print("Warning: Using placeholder build_tcn_model. Replace with your actual model.")
    # input_shape is expected to be (seq_length, features_dim1, features_dim2), e.g., (20, 16, 4)
    seq_length = input_shape[0]
    num_features_per_step = np.prod(input_shape[1:]) # 16 * 4 = 64

    inp = tf.keras.layers.Input(shape=input_shape)

    # Reshape the input to be 3D: (batch_size, seq_length, flattened_features)
    # Target shape for reshape: (seq_length, num_features_per_step) -> (20, 64)
    x = tf.keras.layers.Reshape((seq_length, num_features_per_step))(inp)

    # Example simple structure - REPLACE THIS with your TCN layers
    # Conv1D expects (batch, steps, channels) -> (None, 20, 64)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(x) # Use causal padding for sequence prediction
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # Output layer predicts the flattened next vector
    # Activation is linear (or None) for regression-like prediction of encoded vectors
    out = tf.keras.layers.Dense(num_output_features, activation=None)(x)
    # Reshape the output back to the original target shape (16, 4) if needed for specific loss functions,
    # but MSE can work directly on the flattened vector (num_output_features = 64).
    # If reshaping: out = tf.keras.layers.Reshape(output_shape_dims)(out)

    model = tf.keras.Model(inputs=inp, outputs=out)
    return model
# --- End Placeholder ---


print("Building model...")
# Pass the original input shape and the calculated number of output features
model = build_tcn_model(input_shape, num_output_features)
model.summary() # Print model summary

# Reshape y_train and y_val to match the flattened output of the model
y_train_flat = y_train.reshape(y_train.shape[0], -1)
y_val_flat = y_val.reshape(y_val.shape[0], -1)
print(f"y_train_flat shape: {y_train_flat.shape}, y_val_flat shape: {y_val_flat.shape}")


model.compile(
    optimizer='adam',
    loss='mean_squared_error', # Use MSE for predicting encoded vectors
    metrics=['mae'] # Use Mean Absolute Error as a metric
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(MODELS_DIR, 'tcn_model_best.keras'), save_best_only=True, monitor='val_loss'), # Use .keras format
    tf.keras.callbacks.TensorBoard(log_dir='./logs') # Optional: for TensorBoard visualization
]

# Train model
print("Starting training...")
history = model.fit(
    X_train, y_train_flat, # Use flattened targets
    validation_data=(X_val, y_val_flat), # Use flattened validation targets
    epochs=50,
    batch_size=128,
    callbacks=callbacks
)

# --- Optional: Plot training history ---
print("Training finished. Plotting history...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Loss (Mean Squared Error)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'))
print(f"Training history plot saved to {os.path.join(MODELS_DIR, 'training_history.png')}")
# plt.show() # Uncomment to display plot interactively

print("Script finished.")
