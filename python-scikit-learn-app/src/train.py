import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from data_processing import load_data, clean_data, split_data
from model import Model

def main():
    # Load the data
    data = load_data('data/raw/dataset.csv')  # Update with the actual dataset path
    # Clean the data
    cleaned_data = clean_data(data)
    # Split the data into features and target
    X, y = split_data(cleaned_data)
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = Model()
    model.train(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy:.2f}')
    
    # Save the trained model
    joblib.dump(model, 'model/trained_model.pkl')

if __name__ == '__main__':
    main()