import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm # Optional: for progress bar on large files

# --- Configuration ---
GENERATED_PASSWORDS_FILE = "generated_passwords.txt" # File containing generated passwords
ORIGINAL_DATASETS_DIR = "../Datasets/" # Directory containing original .txt files
MODELS_DIR = "models" # To potentially save analysis results

# Dataset names used for training (must match those used to create the test set)
# This determines which original files to load
DATASET_NAMES = [
    'rockyou'
    # Add other dataset names here if they were part of the training/test split
]

# Splitting parameters (MUST match train.py)
TEST_SPLIT = 0.1
RANDOM_STATE = 42

# --- End Configuration ---

def load_passwords_from_file(filepath):
    """Loads passwords from a file, one password per line."""
    passwords = set()
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                pwd = line.strip()
                if pwd: # Avoid empty lines
                    passwords.add(pwd)
        print(f"Loaded {len(passwords)} unique passwords from {filepath}")
        return passwords
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_password_structure_simple(password):
    """
    Classifies password structure based on character types present.
    Returns a tuple: (has_lower, has_upper, has_digit, has_symbol)
    """
    has_lower = bool(re.search(r'[a-z]', password))
    has_upper = bool(re.search(r'[A-Z]', password))
    has_digit = bool(re.search(r'[0-9]', password))
    # Basic symbol check (anything not letter or digit)
    has_symbol = bool(re.search(r'[^a-zA-Z0-9]', password))
    return (has_lower, has_upper, has_digit, has_symbol)

# --- Main Analysis Logic ---

# 1. Load Generated Passwords
print("--- Loading Generated Passwords ---")
generated_passwords = load_passwords_from_file(GENERATED_PASSWORDS_FILE)
if generated_passwords is None:
    exit(1)
if not generated_passwords:
    print("Warning: No generated passwords loaded. Cannot perform analysis.")
    exit(1)

# 2. Load and Split Original Plain-Text Data to get the Test Set
print("\n--- Loading and Splitting Original Data ---")
all_original_passwords = []
for name in DATASET_NAMES:
    original_file_path = os.path.join(ORIGINAL_DATASETS_DIR, f"{name}.txt")
    print(f"Loading original data from: {original_file_path}")
    try:
        with open(original_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Use tqdm for progress indication on potentially large files
            lines = [line.strip() for line in tqdm(f, desc=f"Reading {name}.txt") if line.strip()]
            all_original_passwords.extend(lines)
            print(f" -> Loaded {len(lines)} passwords.")
    except FileNotFoundError:
        print(f"Error: Original dataset file not found - {original_file_path}")
        # Decide whether to continue or exit if a file is missing
        # continue
        exit(1)
    except Exception as e:
        print(f"Error reading {original_file_path}: {e}")
        exit(1)

if not all_original_passwords:
    print("Error: No original passwords loaded. Cannot create test set.")
    exit(1)

print(f"\nTotal original passwords loaded: {len(all_original_passwords)}")

# Perform the train/test split on the original data to get the test set
# IMPORTANT: Use the exact same test_size and random_state as in train.py
print(f"Splitting original data (Test Size: {TEST_SPLIT}, Random State: {RANDOM_STATE})...")
try:
    _, test_passwords_list = train_test_split(
        all_original_passwords,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE
    )
    test_passwords = set(test_passwords_list) # Convert to set for efficient lookup
    print(f"Created test set with {len(test_passwords)} unique passwords.")
except Exception as e:
    print(f"Error during train_test_split: {e}")
    exit(1)


# 3. Calculate Matching Rate
print("\n--- Calculating Matching Rate ---")
matches = generated_passwords.intersection(test_passwords)
num_matches = len(matches)
num_generated = len(generated_passwords)

matching_rate = num_matches / num_generated if num_generated > 0 else 0
print(f"Generated Passwords: {num_generated}")
print(f"Test Set Passwords: {len(test_passwords)}")
print(f"Exact Matches Found: {num_matches}")
print(f"Matching Rate: {matching_rate:.6f}")


# 4. Calculate Structure Coverage Rate
print("\n--- Calculating Structure Coverage Rate ---")
print("Extracting structures from test set...")
test_structures = set(get_password_structure_simple(pwd) for pwd in tqdm(test_passwords, desc="Test Structures"))
print(f"Found {len(test_structures)} unique structures in the test set.")

print("Extracting structures from generated set...")
generated_structures = set(get_password_structure_simple(pwd) for pwd in tqdm(generated_passwords, desc="Generated Structures"))
print(f"Found {len(generated_structures)} unique structures in the generated set.")

common_structures = test_structures.intersection(generated_structures)
num_common_structures = len(common_structures)
num_test_structures = len(test_structures)

structure_coverage_rate = num_common_structures / num_test_structures if num_test_structures > 0 else 0
print(f"\nCommon Structures Found: {num_common_structures}")
print(f"Structure Coverage Rate: {structure_coverage_rate:.6f} ({num_common_structures} / {num_test_structures})")

# Optional: Save analysis results
# os.makedirs(MODELS_DIR, exist_ok=True)
# with open(os.path.join(MODELS_DIR, "generation_analysis.txt"), "w") as f:
#     f.write(f"Matching Rate: {matching_rate:.6f}\n")
#     f.write(f"Structure Coverage Rate: {structure_coverage_rate:.6f}\n")
#     f.write(f"Num Generated: {num_generated}\n")
#     f.write(f"Num Test: {len(test_passwords)}\n")
#     f.write(f"Num Matches: {num_matches}\n")
#     f.write(f"Num Test Structures: {num_test_structures}\n")
#     f.write(f"Num Generated Structures: {len(generated_structures)}\n")
#     f.write(f"Num Common Structures: {num_common_structures}\n")

print("\nAnalysis script finished.")