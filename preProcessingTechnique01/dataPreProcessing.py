# data_processing.py - Data processing and analysis
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create directories
ASSETS_DIR = "./assets"
PROCESSED_DIR = "./processed_datasets"
MODEL_DIR = "./models"
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"All assets will be saved in {ASSETS_DIR}")
print(f"All processed datasets will be saved in {PROCESSED_DIR}")
print(f"All models will be saved in {MODEL_DIR}")

#############################################
# Data Loading and Analysis Functions
#############################################

def load_dataset(file_path):
    """Load passwords from file with automatic format detection"""
    file_name = os.path.basename(file_path)
    
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                passwords = [line.strip() for line in f if line.strip()]
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            password_col = 'password' if 'password' in df.columns else df.columns[0]
            passwords = df[password_col].dropna().tolist()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        print(f"Loaded {len(passwords):,} passwords from {file_name}")
        return passwords, file_name
        
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return [], file_name

def analyze_dataset(passwords, name, max_length=15, verbose=True):
    """Analyze a password dataset"""
    # Count total passwords
    total_count = len(passwords)
    
    # Filter by length
    filtered_passwords = [pwd for pwd in passwords if len(pwd) <= max_length]
    filtered_count = len(filtered_passwords)
    
    # Handle empty dataset after filtering
    if filtered_count == 0:
        if verbose:
            print(f"No passwords remain in {name} after filtering to ≤{max_length} chars")
        return None
        
    # Calculate length statistics
    lengths = [len(pwd) for pwd in filtered_passwords]
    length_counts = Counter(lengths)
    
    # Categorize passwords
    categories = {
        'alpha_lower': 0,  # Only lowercase letters
        'alpha_mixed': 0,  # Mixed case letters
        'only_numeric': 0, # Only numbers
        'only_special': 0, # Only special chars
        'alphanumeric': 0, # Letters and numbers
        'complex': 0       # Letters, numbers, and special
    }
    
    for pwd in filtered_passwords:
        has_lower = bool(re.search(r'[a-z]', pwd))
        has_upper = bool(re.search(r'[A-Z]', pwd))
        has_digit = bool(re.search(r'[0-9]', pwd))
        has_special = bool(re.search(r'[^a-zA-Z0-9]', pwd))
        
        if has_lower and not has_upper and not has_digit and not has_special:
            categories['alpha_lower'] += 1
        elif (has_lower or has_upper) and not has_digit and not has_special:
            categories['alpha_mixed'] += 1
        elif not has_lower and not has_upper and has_digit and not has_special:
            categories['only_numeric'] += 1
        elif not has_lower and not has_upper and not has_digit and has_special:
            categories['only_special'] += 1
        elif (has_lower or has_upper) and has_digit and not has_special:
            categories['alphanumeric'] += 1
        else:
            categories['complex'] += 1
    
    # Calculate percentages
    categories_pct = {k: 100 * v / filtered_count for k, v in categories.items()}
    
    # Build stats dictionary
    stats = {
        'total': total_count,
        'filtered': filtered_count,
        'filter_pct': 100 * filtered_count / total_count if total_count > 0 else 0,
        'excluded': total_count - filtered_count,
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'std_length': np.std(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'length_counts': length_counts,
        'categories': categories,
        'categories_pct': categories_pct
    }
    
    if verbose:
        print_dataset_stats(name, stats, max_length)
        
    return stats

def print_dataset_stats(name, stats, max_length):
    """Print statistics for a dataset"""
    print(f"\n=== Dataset: {name} ===")
    print(f"Total passwords: {stats['total']:,}")
    print(f"Passwords ≤{max_length} chars: {stats['filtered']:,} ({stats['filter_pct']:.1f}%)")
    print(f"Excluded: {stats['excluded']:,}")
    print(f"\nLength statistics:")
    print(f"  Average: {stats['avg_length']:.2f} chars")
    print(f"  Median: {stats['median_length']:.1f} chars")
    print(f"  Std Dev: {stats['std_length']:.2f}")
    print(f"  Range: {stats['min_length']} to {stats['max_length']} chars")
    
    print(f"\nPassword categories:")
    for cat, count in stats['categories'].items():
        pct = stats['categories_pct'][cat]
        print(f"  {cat.replace('_', ' ').title()}: {count:,} ({pct:.1f}%)")

#############################################
# Password Encoding Functions
#############################################

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
    """Encode a single password into 4D representation"""
    encoded = np.zeros((max_length, 4), dtype=np.float32)

    for i, char in enumerate(password[:max_length]):  # Truncate long passwords
        char_type, serial = CHAR_TYPES.get(char, (4, 0))  # Default: special char
        row, col = ROW_COL_MAP.get(char.lower(), (0, 0))  # Default to (0,0) if unknown

        encoded[i] = [char_type, serial, row, col]

    return encoded

def encode_passwords(passwords, output_file):
    """Encodes a list of passwords and saves as NumPy array."""
    try:
        encoded = np.array([encode_password(p) for p in passwords])
        np.save(output_file, encoded)  # Save as .npy (NumPy format)

        print(f"Encoded {len(passwords)} passwords and saved to {output_file}")
        return encoded
        
    except Exception as e:
        print(f"Error encoding passwords: {e}")
        return None

def decode_password(encoded):
    """Decode a single password from 4D representation"""
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

#############################################
# Data Processing Pipeline
#############################################

def process_password_datasets(dataset_paths, output_dir=PROCESSED_DIR):
    """Process multiple password datasets with encoding"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    all_passwords = []
    processed_datasets = {}
    all_stats = {}
    
    # Process each dataset
    for file_path in dataset_paths:
        # Load passwords
        passwords, dataset_name = load_dataset(file_path)
        if len(passwords) == 0:
            continue
            
        # Analyze dataset
        stats = analyze_dataset(passwords, dataset_name)
        all_stats[dataset_name] = stats
        
        # Remove duplicates
        unique_passwords = list(set(passwords))
        print(f"Removed {len(passwords) - len(unique_passwords)} duplicates")
        
        processed_datasets[dataset_name] = unique_passwords
        
        # Save cleaned passwords
        cleaned_path = os.path.join(output_dir, f"{dataset_name}_cleaned.txt")
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            for p in unique_passwords:
                f.write(p + '\n')
        print(f"Saved cleaned data to {cleaned_path}")
        
        # Add to combined dataset
        all_passwords.extend(unique_passwords)
        
        # Encode passwords
        encoded_path = os.path.join(output_dir, f"{dataset_name}_encoded.npy")
        encode_passwords(unique_passwords, encoded_path)
        
    return processed_datasets, all_stats

def prepare_training_data(encoded_data, seq_length=8, step=1, train_ratio=0.8, val_ratio=0.1):
    """Prepare training, validation and test sets for TCN model"""
    if encoded_data is None:
        return None, None, None
    
    # Fix: ensure encoded_data has correct shape
    if len(encoded_data.shape) == 3:  # [num_passwords, max_length, 4]
        num_passwords, max_length, features = encoded_data.shape
    else:
        print(f"Unexpected shape for encoded_data: {encoded_data.shape}")
        return None, None, None
        
    # Create input-target pairs for sequence prediction
    X, y = [], []
    
    for password_idx in range(num_passwords):
        password_encoded = encoded_data[password_idx]
        
        # Skip empty or padding entries
        if np.all(password_encoded == 0):
            continue
            
        # Find the effective length of the password (non-zero entries)
        effective_length = 0
        for i in range(max_length):
            if not np.all(password_encoded[i] == 0):
                effective_length += 1
            else:
                break
        
        # Skip if password is too short for sequence
        if effective_length < seq_length:
            continue
            
        # Create sequences
        for i in range(effective_length - seq_length + 1):
            X.append(password_encoded[i:i+seq_length-1])  # Input: seq_length-1 characters
            y.append(password_encoded[i+seq_length-1])    # Target: next character
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("No valid sequences could be created. Try using shorter sequence length.")
        return None, None, None
        
    print(f"Created {len(X)} training sequences")
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    
    # Split into train, validation and test sets
    total = len(X)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    
    # Create indices for random splitting
    indices = np.random.permutation(total)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Split data
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

#############################################
# TCN Model Implementation (PyTorch)
#############################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        
        # Padding to maintain the input length
        padding = (kernel_size - 1) * dilation
        
        # First dilated convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        # Second dilated convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        
        # First dilated convolution
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second dilated convolution
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        return x + residual

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, 
                 dropout=0.2, dilation_factor=2):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = dilation_factor ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dilation))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        # Check input dimensions and reshape if needed
        if x.dim() == 4:  # [batch, seq_len-1, max_len, feature_dim]
            batch_size, seq_len, max_len, feature_dim = x.shape
            # Reshape to 3D: [batch * seq_len, max_len, feature_dim]
            x = x.view(-1, max_len, feature_dim)
            
        # Input shape should now be: [batch_size, seq_len, feature_dim]
        # Convert to [batch_size, feature_dim, seq_len] for Conv1D
        x = x.permute(0, 2, 1)
        
        # Apply TCN layers
        x = self.network(x)
        
        # Convert back to [batch_size, seq_len, hidden_dim]
        x = x.permute(0, 2, 1)
        
        # Apply fully connected layer for output
        output = self.fc(x[:, -1, :])  # Use last time step output
        
        return output

#############################################
# Training Functions
#############################################

def train_model(model, train_data, val_data, num_epochs=50, batch_size=128, learning_rate=0.001):
    """Train the TCN model"""
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        # Training
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_X, batch_y in tepoch:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "tcn_best_model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "tcn_best_model.pt")))
    return model

#############################################
# Password Generation and Evaluation
#############################################

def generate_password(model, seed_chars, max_length=16, temperature=1.0):
    """Generate a password by sampling from the model predictions"""
    model.eval()
    
    # Convert seed to tensor
    current_sequence = torch.FloatTensor(seed_chars).unsqueeze(0)  # Add batch dimension
    generated_password = seed_chars.copy()
    
    # Generate characters one by one
    with torch.no_grad():
        for _ in range(max_length - len(seed_chars)):
            # Predict next character
            predictions = model(current_sequence)
            predictions = predictions.view(-1, 4)  # Reshape to 4 dimensions (char_type, serial, row, col)
            
            # Apply temperature to control randomness
            predictions = predictions / temperature
            
            # Sample from the predictions
            next_char = torch.zeros(1, 4)
            for i in range(4):
                # Sample from a uniform distribution between min and max values
                next_char[0, i] = torch.FloatTensor(1).uniform_(0, predictions[0, i].item())
            
            # Append to generated password
            generated_password = np.vstack([generated_password, next_char.numpy()])
            
            # Update current sequence for next iteration
            current_sequence = torch.FloatTensor(generated_password[-current_sequence.size(1):]).unsqueeze(0)
    
    return generated_password

def evaluate_password_strength(model, password, verbose=True):
    """Evaluate password strength using prediction difficulty"""
    model.eval()
    
    # Encode password
    encoded = encode_password(password)
    strength_score = 0
    
    # Get character-by-character prediction difficulty
    with torch.no_grad():
        for i in range(1, len(password)):
            # Get sequence up to current character
            sequence = torch.FloatTensor(encoded[:i]).unsqueeze(0)  # Add batch dimension
            
            # Predict next character
            prediction = model(sequence).numpy()
            
            # Calculate distance between prediction and actual next character
            actual_next = encoded[i]
            distance = np.linalg.norm(prediction - actual_next)
            
            # Add to strength score (higher distance = stronger password)
            strength_score += distance
    
    # Normalize by password length
    normalized_score = strength_score / len(password)
    
    # Convert to 0-100 scale
    final_score = min(100, max(0, normalized_score * 10))
    
    if verbose:
        print(f"Password: {password}")
        print(f"Strength Score: {final_score:.2f}/100")
        print(f"Rating: {'Strong' if final_score > 70 else 'Medium' if final_score > 40 else 'Weak'}")
    
    return final_score

#############################################
# Main Execution
#############################################

def main():
    # Define maximum password length and datasets
    max_length = 16
    datasets = [
        '../datasets/myspace.txt',
        '../datasets/phpbb.txt',
        '../datasets/honeynet.txt',
    ]
    
    # Process datasets
    processed_data, all_stats = process_password_datasets(datasets)
    
    # Load encoded data
    dataset_name = "myspace"  # Choose a specific dataset or use combined
    encoded_data = np.load(os.path.join(PROCESSED_DIR, f"{dataset_name}_encoded.npy"))
    print(f"Loaded {dataset_name} encoded data: {encoded_data.shape}")
    
    # Prepare training data
    train_data, val_data, test_data = prepare_training_data(encoded_data)
    if train_data is None:
        print("Error preparing training data")
        return
    
    # Initialize model
    X_train, y_train = train_data
    input_size = X_train.shape[-1]  # Feature dimension (4)
    output_size = 4  # Predicting 4D character vectors
    
    model = TCN(
        input_size=input_size,
        output_size=output_size,
        num_channels=[64, 128, 256, 128, 64],  # Channel sizes in each layer
        kernel_size=3,
        dropout=0.2
    )
    
    # Train the model
    print("Training model...")
    trained_model = train_model(model, train_data, val_data)
    
    # Test password generation
    print("\nTesting password generation:")
    seed = np.array([[3, 16, 3, 1], [3, 1, 3, 10]])  # "pa" encoded
    generated = generate_password(trained_model, seed)
    decoded = decode_password(generated)
    print(f"Generated password from seed 'pa': {decoded}")
    
    # Test password strength evaluation
    print("\nTesting password strength evaluation:")
    test_passwords = ["password123", "P@ssw0rd!", "qwerty", "xK9#p2L!fR"]
    for pwd in test_passwords:
        evaluate_password_strength(trained_model, pwd)
    
if __name__ == "__main__":
    main()

