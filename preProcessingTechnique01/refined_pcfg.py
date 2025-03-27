import re
from collections import Counter
import math

# Predefined website suffixes
WEBSITE_SUFFIXES = [".com", ".net", ".org", ".gov", ".edu", ".cn", ".uk"]

def recognize_years(password):
    """Extracts 4-digit years between 1900-2100 from the password."""
    match = re.findall(r'\b(19\d{2}|20[01]\d|2100)\b', password)
    return match if match else []

def recognize_websites(password):
    """Extracts website names based on common suffixes."""
    for suffix in WEBSITE_SUFFIXES:
        if suffix in password:
            website_segment = password.split(suffix)[0] + suffix
            return website_segment
    return None

def recognize_keyboard_patterns(password):
    """Identifies common keyboard patterns with multiple character types."""
    keyboard_patterns = [
        r"qwerty", r"asdfgh", r"zxcvbn",  # Horizontal keyboard patterns
        r"qazwsx", r"wsxedc", r"edcrfv",  # Vertical keyboard patterns
        r"1qaz", r"2wsx", r"3edc"  # Diagonal keyboard patterns
    ]
    
    for pattern in keyboard_patterns:
        if re.search(pattern, password, re.IGNORECASE):
            return pattern
    return None  

def recognize_words(password, word_dict):
    """Checks if a segment in the password matches a word in the dictionary."""
    words_found = [word for word in word_dict if word in password]
    return words_found

def build_word_dictionary(passwords, min_length=3, min_frequency=5):
    """
    Constructs a word dictionary from a list of passwords using frequency filtering.
    Uses Algorithm 1 from the research paper.
    """
    word_counts = Counter()
    
    for password in passwords:
        letter_segments = re.findall(r"[a-zA-Z]+", password)  # Extract letter-only segments
        for segment in letter_segments:
            if len(segment) >= min_length:
                word_counts[segment] += 1  # Count frequency
    
    # Filter words based on minimum frequency
    word_dict = {word for word, count in word_counts.items() if count >= min_frequency}
    return word_dict

def zipf_distribution_check(frequencies):
    """
    Validates if the extracted segments follow a Zipfian distribution.
    Implements equation: log(fr) = log(C) - s * log(r)
    """
    sorted_freqs = sorted(frequencies.values(), reverse=True)
    ranks = list(range(1, len(sorted_freqs) + 1))

    log_freqs = [math.log(f) for f in sorted_freqs]
    log_ranks = [math.log(r) for r in ranks]

    # Check linear relationship
    correlation = sum(f * r for f, r in zip(log_freqs, log_ranks)) / len(log_freqs)
    return correlation

def refined_pcfg_segmentation(password, word_dict):
    """Segments the password using the refined PCFG approach."""
    segments = {}
    
    segments["years"] = recognize_years(password)
    segments["website"] = recognize_websites(password)
    segments["keyboard_pattern"] = recognize_keyboard_patterns(password)
    segments["words"] = recognize_words(password, word_dict)
    
    return segments

# Example usage
if __name__ == "__main__":
    sample_passwords = ["1998mypassword@facebook.com", "q1w2e3r4", "john2001@csdn.net"]
    
    # Build dictionary from dataset
    word_dict = build_word_dictionary(sample_passwords)
    
    for password in sample_passwords:
        print(f"Password: {password}")
        print(refined_pcfg_segmentation(password, word_dict))
        print()

