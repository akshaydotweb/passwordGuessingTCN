import os
import numpy as np
import refined_pcfg  
import encode_4d  

def load_and_clean_passwords(file_path):
    """Loads a password file, removes duplicates, and cleans text."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        passwords = f.readlines()
    
    passwords = [p.strip() for p in passwords if p.strip()]  
    passwords = list(set(passwords))  # Remove duplicates

    print(f"Loaded {file_path}: {len(passwords)} unique passwords")
    return passwords

def save_cleaned_data(passwords, output_file):
    """Saves cleaned passwords to a new file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for p in passwords:
            f.write(p + '\n')
    print(f"Saved cleaned data to {output_file}")

# Load and clean both datasets
honeynet_cleaned = load_and_clean_passwords("honeynet.txt")
myspace_cleaned = load_and_clean_passwords("myspace.txt")

# Save cleaned versions
save_cleaned_data(honeynet_cleaned, "honeynet_cleaned.txt")
save_cleaned_data(myspace_cleaned, "myspace_cleaned.txt")


#  **Encode cleaned passwords using 4D encoding**
encode_4d.encode_passwords("honeynet_cleaned.txt", "honeynet_encoded.npy")
encode_4d.encode_passwords("myspace_cleaned.txt", "myspace_encoded.npy")


#  **Load encoded passwords**
honeynet_encoded = np.load("honeynet_encoded.npy")
myspace_encoded = np.load("myspace_encoded.npy")


# Fix: Use CHAR_TYPES from encode_4d.py
def decode_password(encoded_password):
    """Decodes a 4D encoded password back to a text string."""
    char_map = {v: k for k, v in encode_4d.CHAR_TYPES.items()}  # Fix applied
    return ''.join(char_map.get((row, col), '') for row, col in encoded_password[:, :2])

honeynet_decoded = ["".join(decode_password(p)) for p in honeynet_encoded]
myspace_decoded = ["".join(decode_password(p)) for p in myspace_encoded]

# **Build word dictionary from decoded passwords**
all_passwords = honeynet_decoded + myspace_decoded
word_dict = refined_pcfg.build_word_dictionary(all_passwords)  

def test_on_dataset(passwords, output_path, word_dict):
    """Applies refined PCFG segmentation on passwords and saves output."""
    with open(output_path, "w", encoding="utf-8") as outfile:
        for password in passwords:
            segmented = refined_pcfg.refined_pcfg_segmentation(password, word_dict)  
            outfile.write(f"{password} → {segmented}\n")
    
    print(f"Processed passwords → Output saved in {output_path}")

# Apply refined PCFG segmentation on decoded passwords
test_on_dataset(honeynet_decoded, "honeynet_segmented.txt", word_dict)
test_on_dataset(myspace_decoded, "myspace_segmented.txt", word_dict)





