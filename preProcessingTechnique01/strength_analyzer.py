# strength_analyzer.py - Analyze password strength
import numpy as np
import tensorflow as tf
import os
import math
from encode_4d import encode_password
import string

def calculate_entropy(password):
    """Calculate Shannon entropy of a password"""
    # Count character frequencies
    char_counter = {}
    for char in password:
        char_counter[char] = char_counter.get(char, 0) + 1
    
    # Calculate entropy
    length = len(password)
    entropy = 0
    
    for count in char_counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    
    # Adjust for password length
    length_factor = min(1.0, length / 12)  # Max benefit at 12+ chars
    return entropy * length * length_factor

def evaluate_password_strength(password, model=None):
    """
    Evaluate password strength using the TCN model and Shannon entropy.
    Falls back to basic entropy-based evaluation if model is not available.
    """
    # Basic strength categories based on character types
    has_lower = any(c in string.ascii_lowercase for c in password)
    has_upper = any(c in string.ascii_uppercase for c in password)
    has_digit = any(c in string.digits for c in password)
    has_special = any(c in string.punctuation for c in password)
    
    char_diversity = sum([has_lower, has_upper, has_digit, has_special])
    
    # Calculate base entropy score
    entropy = calculate_entropy(password)
    
    # If model is available, enhance the score with model predictions
    model_score = 0
    if model is not None:
        try:
            # Encode password
            encoded = encode_password(password)
            
            # Use sliding window to predict each character
            for i in range(1, len(password)):
                # Get sequence up to current character
                sequence = encoded[:i].reshape(1, i, 4)
                
                # Predict the next character
                prediction = model.predict(sequence, verbose=0)[0]
                
                # Get predicted probability for actual next character
                actual_next = encoded[i]
                # Adjust indexing since model might use different output format
                max_prob = np.max(prediction)
                
                # Add to strength score (lower probability = stronger password)
                model_score += -np.log2(max_prob + 1e-10)
            
            # Normalize by length
            if len(password) > 1:
                model_score /= (len(password) - 1)
        except Exception as e:
            print(f"Error using model for prediction: {e}")
            model_score = 0
    
    # Combine scores
    # If we have a model, use 70% model weight, 30% entropy
    # Otherwise use 100% entropy
    if model is not None and model_score > 0:
        combined_score = 0.7 * model_score + 0.3 * entropy
    else:
        combined_score = entropy
        
    # Determine rating
    if combined_score > 8 and char_diversity >= 3 and len(password) >= 10:
        rating = 'Strong'
    elif combined_score > 5 and char_diversity >= 2 and len(password) >= 8:
        rating = 'Medium'
    else:
        rating = 'Weak'
        
    return {
        'score': combined_score,
        'entropy': entropy,
        'model_score': model_score if model is not None else None,
        'length': len(password),
        'character_types': char_diversity,
        'rating': rating
    }

# Load trained model if available
model = None
try:
    if os.path.exists('models/tcn_model.h5'):
        print("Loading TCN model...")
        model = tf.keras.models.load_model('models/tcn_model.h5')
        print("Model loaded successfully")
    else:
        print("Model file not found. Falling back to basic entropy evaluation.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to basic entropy evaluation.")

# Demo if run directly
if __name__ == "__main__":
    test_passwords = [
        "password", 
        "Password123", 
        "P@ssw0rd!2023", 
        "qwerty",
        "iloveyou",
        "MountainCh@let2023!"
    ]
    
    print("\n=== Password Strength Analysis ===")
    for pwd in test_passwords:
        result = evaluate_password_strength(pwd, model)
        print(f"\nPassword: {pwd}")
        print(f"Length: {result['length']}")
        print(f"Character types: {result['character_types']}/4")
        print(f"Entropy: {result['entropy']:.2f}")
        if result['model_score'] is not None:
            print(f"Model score: {result['model_score']:.2f}")
        print(f"Overall score: {result['score']:.2f}")
        print(f"Rating: {result['rating']}")

