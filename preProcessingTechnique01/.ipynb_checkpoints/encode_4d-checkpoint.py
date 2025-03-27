import numpy as np
import string

# Define character types
CHAR_TYPES = {
    **{c: (1, i + 1) for i, c in enumerate(string.digits)},         # Digits
    **{c: (2, i + 1) for i, c in enumerate(string.ascii_uppercase)}, # Uppercase
    **{c: (3, i + 1) for i, c in enumerate(string.ascii_lowercase)}, # Lowercase
    **{c: (4, i + 1) for i, c in enumerate(string.punctuation)}       # Special
}

# Define QWERTY keyboard layout for row/column mapping
QWERTY_LAYOUT = [
    "`1234567890-=",
    "qwertyuiop[]\\",
    "asdfghjkl;'",
    "zxcvbnm,./"
]
ROW_COL_MAP = {char: (row_idx + 1, col_idx + 1)
               for row_idx, row in enumerate(QWERTY_LAYOUT)
               for col_idx, char in enumerate(row)}

# Function to encode a password using the 4D method
def encode_password(password, max_length=16):
    """Converts a password into a 4D numerical vector representation."""
    encoded = np.zeros((max_length, 4), dtype=np.float32)

    for i, char in enumerate(password[:max_length]):  # Truncate long passwords
        char_type, serial = CHAR_TYPES.get(char, (4, 0))  # Default: special char
        row, col = ROW_COL_MAP.get(char, (0, 0))  # Default to (0,0) if unknown

        encoded[i] = [char_type, serial, row, col]

    return encoded

# Function to process and save encoded passwords
def encode_passwords(file_path, output_file):
    """Reads passwords, encodes them, and saves as NumPy array."""
    with open(file_path, 'r', encoding='utf-8') as f:
        passwords = [line.strip() for line in f.readlines()]

    encoded = np.array([encode_password(p) for p in passwords])
    np.save(output_file, encoded)  # Save as .npy (NumPy format)

    print(f"Encoded {len(passwords)} passwords and saved to {output_file}")

# Encode both datasets
encode_passwords("honeynet_cleaned.txt", "honeynet_encoded.npy")
encode_passwords("myspace_cleaned.txt", "myspace_encoded.npy")

