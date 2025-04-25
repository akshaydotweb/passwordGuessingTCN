# password_generator.py - Generate and evaluate passwords
import numpy as np
import tensorflow as tf
# Make sure encode_4d is accessible and has decode_password
from encode_4d import decode_password, encode_password # Import encode_password as well

# --- Configuration ---
MODEL_PATH = 'models/tcn_model_best.keras'
# This MUST match the dimensions used in encoding/training (e.g., (16, 4))
# It's the shape of ONE encoded character vector in the sequence.
# The model predicts the *next* vector of this shape (flattened).
ORIGINAL_VECTOR_DIMS = (16, 4)
# --- End Configuration ---


# Load the model
print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary() # Print summary to verify layers
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Get expected sequence length from the loaded model's input shape
# Input shape is (None, seq_len, dim1, dim2) -> e.g., (None, 20, 16, 4)
MODEL_INPUT_SEQ_LEN = model.input_shape[1]
# Verify the model's input shape matches the expected vector dimensions
if model.input_shape[2:] != ORIGINAL_VECTOR_DIMS:
    print(f"Warning: Model input shape {model.input_shape[2:]} does not match configured ORIGINAL_VECTOR_DIMS {ORIGINAL_VECTOR_DIMS}")
    # Decide how to handle this - maybe exit or try to proceed cautiously
    # exit(1)

# Calculate the expected flattened output size
EXPECTED_FLAT_OUTPUT_SIZE = np.prod(ORIGINAL_VECTOR_DIMS)
if model.output_shape[1] != EXPECTED_FLAT_OUTPUT_SIZE:
     print(f"Warning: Model output size {model.output_shape[1]} does not match expected flattened size {EXPECTED_FLAT_OUTPUT_SIZE} from ORIGINAL_VECTOR_DIMS {ORIGINAL_VECTOR_DIMS}")
     # exit(1)


def generate_password_mse(model, seed_sequence_vectors, max_generated_length=16):
    """
    Generates a sequence of encoded vectors using a model trained with MSE loss.

    Args:
        model: The loaded Keras model.
        seed_sequence_vectors: A numpy array of seed vectors, shape (seed_len, dim1, dim2).
                               The length `seed_len` can be less than the model's required input length.
        max_generated_length: The maximum total length of the generated sequence (including seed).

    Returns:
        A numpy array containing the sequence of generated vectors (including the seed),
        shape (total_length, dim1, dim2).
    """
    if seed_sequence_vectors.ndim != 3 or seed_sequence_vectors.shape[1:] != ORIGINAL_VECTOR_DIMS:
        raise ValueError(f"Seed sequence must have shape (seed_len, {ORIGINAL_VECTOR_DIMS[0]}, {ORIGINAL_VECTOR_DIMS[1]}), but got {seed_sequence_vectors.shape}")

    current_seed_len = len(seed_sequence_vectors)
    # Ensure seed vectors are float32, as expected by the model
    generated_vectors = list(seed_sequence_vectors.astype(np.float32)) # Start with the seed vectors

    print(f"Starting generation with seed length: {current_seed_len}, target total length: {max_generated_length}")

    # Pad the initial sequence if it's shorter than the model's input requirement
    if current_seed_len < MODEL_INPUT_SEQ_LEN:
        padding_needed = MODEL_INPUT_SEQ_LEN - current_seed_len
        # Use zero padding consistent with training (assuming float32)
        padding_vector = np.zeros(ORIGINAL_VECTOR_DIMS, dtype=np.float32)
        padding = [padding_vector] * padding_needed
        # Prepend padding
        current_input_sequence_list = padding + generated_vectors
        print(f"Padded seed sequence with {padding_needed} vectors.")
    else:
        # Take the last MODEL_INPUT_SEQ_LEN vectors if the seed is longer
        current_input_sequence_list = generated_vectors[-MODEL_INPUT_SEQ_LEN:]

    # Ensure current_input_sequence_list has the correct length before starting loop
    if len(current_input_sequence_list) != MODEL_INPUT_SEQ_LEN:
         raise RuntimeError(f"Internal error before loop: current_input_sequence length is {len(current_input_sequence_list)}, expected {MODEL_INPUT_SEQ_LEN}")

    # Generate new vectors one by one
    while len(generated_vectors) < max_generated_length:
        # Prepare input: shape (1, seq_len, dim1, dim2)
        input_for_prediction = np.array([current_input_sequence_list], dtype=np.float32) # Ensure float32 input

        # Predict the *next* vector (model output is flattened)
        predicted_flat_vector = model.predict(input_for_prediction, verbose=0)[0] # verbose=0 to avoid progress bar per step

        # Reshape the predicted flat vector back to the original vector shape
        try:
            next_vector_continuous = predicted_flat_vector.reshape(ORIGINAL_VECTOR_DIMS)
        except ValueError as e:
            print(f"Error reshaping predicted vector (shape {predicted_flat_vector.shape}) to {ORIGINAL_VECTOR_DIMS}: {e}")
            print("Cannot continue generation.")
            break # Exit the generation loop

        # --- Discretization Step ---
        # Convert continuous MSE output to discrete integer values suitable for decoding.
        next_vector_discrete = np.round(next_vector_continuous).astype(int)
        # Clip values to the expected range (e.g., 0-15 for 4-bit encoding components)
        # Adjust bounds if your encoding uses a different range
        next_vector_discrete = np.clip(next_vector_discrete, 0, max(15, np.max(next_vector_discrete))) # Clip based on typical encoding or observed max

        # Append the processed vector (as float32 for consistency in the list)
        # to the overall generated sequence (for final output)
        generated_vectors.append(next_vector_discrete.astype(np.float32))

        # Update the input sequence list for the next prediction: slide the window
        # Remove the oldest vector (at the beginning)
        current_input_sequence_list.pop(0)
        # Append the newly generated vector (at the end)
        current_input_sequence_list.append(next_vector_discrete.astype(np.float32))

    print(f"Finished generation. Total vectors generated: {len(generated_vectors)}")
    # Return the final sequence as integers, suitable for decoding
    return np.array(generated_vectors).astype(int)


# --- Example Usage ---
# Define a seed string
seed_string = "pa"
print(f"Using seed string: '{seed_string}'")

# Encode the seed string using the imported function
# encode_password returns shape (max_length, 4), we only need the actual encoded chars
encoded_seed_full = encode_password(seed_string)
# Extract only the non-zero vectors corresponding to the seed string length
seed_input_vectors = encoded_seed_full[:len(seed_string)] # Shape: (len(seed_string), 16, 4)

if len(seed_input_vectors) == 0 and len(seed_string) > 0:
     print(f"Warning: Encoding the seed string '{seed_string}' resulted in zero vectors. Check encoding logic.")
     # Fallback to a single zero vector if encoding fails, though generation might be poor
     seed_input_vectors = np.zeros((1,) + ORIGINAL_VECTOR_DIMS, dtype=np.float32)
elif len(seed_input_vectors) > 0:
     print(f"Encoded seed input vectors shape: {seed_input_vectors.shape}")
else: # Handle empty seed string case
     print("Seed string is empty. Starting generation from scratch (using padding).")
     # Provide an empty array, generate_password_mse will handle padding
     seed_input_vectors = np.empty((0,) + ORIGINAL_VECTOR_DIMS, dtype=np.float32)


# Define max_generated_length in this scope
max_generated_length = 10 # Target total length (seed + generated)

# Generate using the function
generated_vectors_output = generate_password_mse(model, seed_input_vectors, max_generated_length=max_generated_length) # Pass the variable

print(f"Generated vectors output shape: {generated_vectors_output.shape}")

# --- Print the generated vectors ---
print("Generated Vectors (rounded & clipped integers):")
# Limit printing very large arrays for readability
print(generated_vectors_output[:min(max_generated_length, len(generated_vectors_output))])
# --- End Print Vectors ---

# Decode the generated vectors
# The quality depends heavily on the model and the discretization method.
try:
    # Assuming decode_password takes the full sequence of integer vectors
    decoded_password_str = decode_password(generated_vectors_output)
    print(f"Attempted decoded password (plain text): '{decoded_password_str}'") # Added label
except ImportError:
    print("Could not import 'decode_password' from 'encode_4d'. Skipping decoding.")
except Exception as e:
    print(f"Error during decoding: {e}")
    print("This might happen if the generated vectors (after rounding/clipping) are not valid according to the encoding scheme.")

print("Script finished.")