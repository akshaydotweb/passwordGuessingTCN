# password_generator.py - Generate and evaluate passwords
import numpy as np
import tensorflow as tf
from tqdm import tqdm # Import tqdm
import string # Import string for character sets
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
        seed_sequence_vectors: A numpy array of seed vectors, shape (seed_len, 16, 4).
                               The length `seed_len` can be less than the model's required input length.
        max_generated_length: The maximum total length of the generated sequence (including seed).

    Returns:
        A numpy array containing the sequence of generated vectors (including the seed),
        shape (total_length, 16, 4), dtype=int.
    """
    # Input validation (shape check)
    if seed_sequence_vectors.ndim != 3 or (seed_sequence_vectors.shape[0] > 0 and seed_sequence_vectors.shape[1:] != ORIGINAL_VECTOR_DIMS):
        # Check if it's the common error case from incorrect encoding
        if seed_sequence_vectors.ndim == 2 and seed_sequence_vectors.shape[0] > 0:
             print(f"Error: Seed sequence has shape {seed_sequence_vectors.shape}, but expected (seed_len, {ORIGINAL_VECTOR_DIMS[0]}, {ORIGINAL_VECTOR_DIMS[1]}).")
             print("       This likely means encode_password in encode_4d.py is returning the wrong shape.")
             raise ValueError(f"Incorrect seed shape from encode_password: got {seed_sequence_vectors.shape}, expected 3D.")
        # Handle the case of an empty seed (shape (0, 16, 4)) which is valid
        elif seed_sequence_vectors.shape == (0,) + ORIGINAL_VECTOR_DIMS:
             pass # Empty seed is okay, will be padded
        else:
             raise ValueError(f"Seed sequence must have shape (seed_len, {ORIGINAL_VECTOR_DIMS[0]}, {ORIGINAL_VECTOR_DIMS[1]}), but got {seed_sequence_vectors.shape}")


    current_seed_len = len(seed_sequence_vectors)
    # Ensure seed vectors are float32, as expected by the model
    generated_vectors = list(seed_sequence_vectors.astype(np.float32)) # Start with the seed vectors

    # Pad the initial sequence if it's shorter than the model's input requirement
    if current_seed_len < MODEL_INPUT_SEQ_LEN:
        padding_needed = MODEL_INPUT_SEQ_LEN - current_seed_len
        padding_vector = np.zeros(ORIGINAL_VECTOR_DIMS, dtype=np.float32)
        padding = [padding_vector] * padding_needed
        current_input_sequence_list = padding + generated_vectors
    else:
        current_input_sequence_list = generated_vectors[-MODEL_INPUT_SEQ_LEN:]

    if len(current_input_sequence_list) != MODEL_INPUT_SEQ_LEN:
         raise RuntimeError(f"Internal error before loop: current_input_sequence length is {len(current_input_sequence_list)}, expected {MODEL_INPUT_SEQ_LEN}")

    # Generate new vectors one by one
    while len(generated_vectors) < max_generated_length:
        input_for_prediction = np.array([current_input_sequence_list], dtype=np.float32)
        predicted_flat_vector = model.predict(input_for_prediction, verbose=0)[0]

        try:
            next_vector_continuous = predicted_flat_vector.reshape(ORIGINAL_VECTOR_DIMS)
        except ValueError as e:
            break # Exit loop if reshape fails

        next_vector_discrete = np.round(next_vector_continuous).astype(int)
        # *** Recommendation: Use a fixed upper bound if known ***
        next_vector_discrete = np.clip(next_vector_discrete, 0, 15) # Assuming 0-15 is the correct range

        generated_vectors.append(next_vector_discrete.astype(np.float32))
        current_input_sequence_list.pop(0)
        current_input_sequence_list.append(next_vector_discrete.astype(np.float32))

    return np.array(generated_vectors).astype(int)


# --- Bulk Generation ---
def bulk_generate_and_save(model, num_passwords, max_length, output_file):
    """Generates multiple passwords using random seeds and saves them to a file."""
    print(f"Starting bulk generation of {num_passwords} passwords (max length {max_length})...")
    generated_passwords = []

    # Define a character set for generating random seeds
    # Adjust this based on characters expected/trained on
    seed_char_set = list(string.ascii_lowercase + string.digits) # Example: lowercase + digits
    # Or use a broader set: list(string.ascii_letters + string.digits + string.punctuation)

    for _ in tqdm(range(num_passwords), desc="Generating Passwords"):
        try:
            # --- Create a realistic random seed ---
            seed_len = np.random.randint(1, 5) # Generate seeds of length 1 to 4
            seed_chars = np.random.choice(seed_char_set, seed_len)
            seed_string = "".join(seed_chars)
            # --- End create seed ---

            # Encode the generated seed string
            # IMPORTANT: Assumes encode_password returns correct shape (len, 16, 4)
            encoded_seed_full = encode_password(seed_string)
            seed_input_vectors = encoded_seed_full[:len(seed_string)]

            # Handle potential empty result from encoding (though less likely with non-empty seed)
            if len(seed_input_vectors) == 0 and len(seed_string) > 0:
                 print(f"\nWarning: Encoding seed '{seed_string}' failed. Skipping.")
                 continue # Skip this iteration

            # Generate the rest of the password
            generated_vectors = generate_password_mse(model, seed_input_vectors, max_generated_length=max_length)

            # Decode the generated vectors
            decoded_password = decode_password(generated_vectors)
            if decoded_password: # Only add non-empty passwords
                generated_passwords.append(decoded_password)

        except ValueError as ve:
             # Catch potential shape errors from generate_password_mse if encode_password is wrong
             print(f"\nSkipping generation due to ValueError (likely encode_password shape issue for seed '{seed_string}'): {ve}")
             continue
        except Exception as e:
            # Catch other errors during generation/decoding for one password
            print(f"\nError during generation/decoding for seed '{seed_string}': {e}")
            continue # Skip this password and continue

    print(f"\nGenerated {len(generated_passwords)} valid passwords.")

    # Save the generated passwords to the specified file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for pwd in generated_passwords:
                f.write(pwd + '\n')
        print(f"Successfully saved generated passwords to {output_file}")
    except Exception as e:
        print(f"Error saving generated passwords to {output_file}: {e}")

# --- Example Usage (Single Generation - Optional, can be commented out) ---
# ... (keep the single generation example if useful for debugging) ...
# print(f"--- Single Generation Example ---")
# ... etc ...
# print(f"--- End Single Generation Example ---")


# --- Run Bulk Generation ---
NUM_TO_GENERATE = 1000 # Number of passwords to generate
MAX_GEN_LENGTH_BULK = 16 # Maximum length for bulk generated passwords
OUTPUT_FILENAME = "generated_passwords.txt"

bulk_generate_and_save(model, NUM_TO_GENERATE, MAX_GEN_LENGTH_BULK, OUTPUT_FILENAME)

print("\nScript finished.")