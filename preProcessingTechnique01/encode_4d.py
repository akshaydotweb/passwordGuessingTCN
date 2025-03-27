import numpy as np
import string
import os

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

def encode_password(password, max_length=16):

    encoded = np.zeros((max_length, 4), dtype=np.float32)

    for i, char in enumerate(password[:max_length]):  # Truncate long passwords
        char_type, serial = CHAR_TYPES.get(char, (4, 0))  # Default: special char
        row, col = ROW_COL_MAP.get(char, (0, 0))  # Default to (0,0) if unknown

        encoded[i] = [char_type, serial, row, col]

    return encoded

def encode_passwords(file_path, output_file):
    """
    Reads passwords from file, encodes them, and saves as NumPy array.
    
    Args:
        file_path: Path to password text file (one password per line)
        output_file: Path to save encoded passwords (.npy format)
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found!")
            return False
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            passwords = [line.strip() for line in f.readlines() if line.strip()]

        encoded = np.array([encode_password(p) for p in passwords])
        np.save(output_file, encoded)  # Save as .npy (NumPy format)

        print(f"Encoded {len(passwords)} passwords and saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error encoding passwords: {e}")
        return False

def decode_password(encoded):

    decoded = []
    
    # Create reverse mappings
    reverse_map = {}
    for char, (char_type, serial) in CHAR_TYPES.items():
        reverse_map[(char_type, serial)] = char
    
    for i in range(encoded.shape[0]):
        char_vec = encoded[i]
        if np.all(char_vec == 0):  # End of password
            break
            
        char_type, serial = int(char_vec[0]), int(char_vec[1])
        if (char_type, serial) in reverse_map:
            decoded.append(reverse_map[(char_type, serial)])
        else:
            decoded.append('?')  # Unknown character
            
    return ''.join(decoded)

# Only run the examples if the script is executed directly
if __name__ == "__main__":
    # Example usage
    test_password = "a1"
    encoded = encode_password(test_password)
    print(f"Original: {test_password}")
    print(f"Encoded: {encoded}")
    decoded = decode_password(encoded)
    print(f"Decoded: {decoded}")


