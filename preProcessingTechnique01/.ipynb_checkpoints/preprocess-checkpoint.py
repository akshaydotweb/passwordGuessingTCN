import os
import numpy as np
import refined_pcfg
import encode_4d

def process_password_datasets(dataset_paths, output_dir="processed_datasets", 
                             clean=True, encode=True, segment=True):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    all_passwords = []
    processed_datasets = {}
    
    # Process each dataset
    for file_path in dataset_paths:
        dataset_name = os.path.basename(file_path).split('.')[0]
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Load and clean passwords
        passwords = load_and_clean_passwords(file_path)
        processed_datasets[dataset_name] = passwords
        
        if clean:
            # Save cleaned passwords
            cleaned_path = os.path.join(output_dir, f"{dataset_name}_cleaned.txt")
            save_cleaned_data(passwords, cleaned_path)
            
        # Add to combined dataset for dictionary building
        all_passwords.extend(passwords)
        
        if encode:
            # Encode passwords
            encoded_path = os.path.join(output_dir, f"{dataset_name}_encoded.npy")
            encode_4d.encode_passwords(
                output_dir + f"/{dataset_name}_cleaned.txt" if clean else file_path, 
                encoded_path
            )
    
    # Build word dictionary from all passwords for segmentation
    if segment:
        word_dict = refined_pcfg.build_word_dictionary(all_passwords)
        
        # Segment each dataset
        for dataset_name, passwords in processed_datasets.items():
            segmented_path = os.path.join(output_dir, f"{dataset_name}_segmented.txt")
            test_on_dataset(passwords, segmented_path, word_dict)
    
    return processed_datasets

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

def test_on_dataset(passwords, output_path, word_dict):
    """Applies refined PCFG segmentation on passwords and saves output."""
    with open(output_path, "w", encoding="utf-8") as outfile:
        for password in passwords:
            segmented = refined_pcfg.refined_pcfg_segmentation(password, word_dict)  
            outfile.write(f"{password} → {segmented}\n")
    
    print(f"Processed passwords → Output saved in {output_path}")

# Usage example
if __name__ == "__main__":
    # Define dataset paths
    datasets = [
        "../datasets/myspace.txt",
        "../datasets/honeynet.txt", 
        "../datasets/example.txt"
    ]
    
    # Process all datasets
    processed_data = process_password_datasets(datasets)





