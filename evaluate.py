import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_simulated_data(file_path):
    with open(file_path, 'r') as file:
        simulated_data = json.load(file)
    return simulated_data

def preprocess_data(data):
    df = pd.DataFrame(data)
    purchase_sequences = [entry['purchases'] for entry in data]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(purchase_sequences)
    encoded_sequences = tokenizer.texts_to_sequences(purchase_sequences)
    max_sequence_len = max(len(seq) for seq in encoded_sequences)
    padded_sequences = pad_sequences(encoded_sequences, maxlen=max_sequence_len, padding='pre')
    return df, tokenizer, padded_sequences, max_sequence_len

if __name__ == "__main__":
    file_path = 'D:\\AIML Assignment\\simulated_purchase_history.json'
    
    # Load and preprocess data
    data = load_simulated_data(file_path)
    df, tokenizer, padded_sequences, max_sequence_len = preprocess_data(data)
    
    # Preparing predictors and label sequences
    X = padded_sequences[:, :-1]
    y = padded_sequences[:, -1]
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Load the model
    model = tf.keras.models.load_model('purchase_prediction_model.h5')
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
