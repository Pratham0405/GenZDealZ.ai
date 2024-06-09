import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import json
import pandas as pd

# Ensure you have loaded and preprocessed the data correctly

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

def build_model(input_dim, embedding_size, input_length):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_size, input_length=input_length))
    model.add(SimpleRNN(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(SimpleRNN(100))
    model.add(Dropout(0.3))
    model.add(Dense(input_dim, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

if __name__ == "__main__":
    file_path = 'simulated_purchase_history.json'
    
    # Load and preprocess data
    data = load_simulated_data(file_path)
    df, tokenizer, padded_sequences, max_sequence_len = preprocess_data(data)
    
    # Preparing predictors and label sequences
    X = padded_sequences[:, :-1]
    y = padded_sequences[:, -1]
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Building the model
    embedding_size = 20
    input_dim = len(tokenizer.word_index) + 1
    input_length = max_sequence_len - 1
    model = build_model(input_dim, embedding_size, input_length)
    
    # Training the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model
    model.save('purchase_prediction_model.h5')
    
    print("Model training completed and saved.")
