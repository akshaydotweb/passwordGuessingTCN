# password_generator.py - Generate and evaluate passwords
import numpy as np
import tensorflow as tf
from encode_4d import decode_password

# Load the model
model = tf.keras.models.load_model('models/tcn_model.h5')

def generate_password(seed_chars, max_length=16, temperature=1.0):
    """Generate a password by sampling from the model predictions"""
    current_sequence = np.array([seed_chars])
    generated_password = seed_chars.copy()
    
    for _ in range(max_length - len(seed_chars)):
        # Predict next character
        predictions = model.predict(current_sequence)[0]
        
        # Apply temperature to control randomness
        predictions = np.log(predictions) / temperature
        exp_predictions = np.exp(predictions)
        predictions = exp_predictions / np.sum(exp_predictions)
        
        # Sample from the distribution
        next_char = np.zeros((1, 4))
        for i in range(4):
            next_char[0, i] = np.random.choice(range(predictions.shape[i]), 
                                             p=predictions[:, i])
        
        # Update sequences
        generated_password = np.vstack([generated_password, next_char])
        current_sequence = np.array([generated_password[-model.input_shape[1]:]])
        
    return generated_password

# Example usage
seed = np.array([[3, 16, 3, 1], [3, 1, 3, 10]])  # "pa" encoded
generated = generate_password(seed)
decoded = decode_password(generated)
print(f"Generated password: {decoded}")