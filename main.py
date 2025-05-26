import json
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the Decision Logic model
filename_decision = 'dl_model.pkl'
nn_model = pickle.load(open(filename_decision, 'rb'))

# Load the RNN model
model = keras.models.load_model('rnn_model.h5')

# Load the label encoder and vectorizer
label_encoder = LabelEncoder()  
vectorizer = TfidfVectorizer() 
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)

# Category keywords
category_keywords = { 
    "Reparasi atap": ["genteng", "atap", "bocor", "tetesan", "plafon", "baja ringan", "asbes"],
    "Reparasi saluran air": ["saluran", "air", "pipa", "mampet", "keran", "saluran wastafel", "septictank", "water heater"],
    "Reparasi lantai dan dinding": ["dinding", "lantai", "keramik", "retak", "pecah", "cat", "dinding lembab", "wallpaper", "marmer"],
    "Instalasi listrik": ["kabel", "listrik", "lampu", "arus", "sambungan", "instalasi CCTV", "instalasi WiFi", "genset", "instalasi AC"],
    "Reparasi aksesoris": ["pintu", "jendela", "meja", "kursi", "pagar", "kusen", "gorden", "alumunium"]
}

# Function for combined prediction
def combined_predict(text, model, tokenizer, label_encoder, category_keywords):
    # Tokenize and vectorize text for model prediction
    text_vectorized = tokenizer.texts_to_sequences([text])
    text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_vectorized, padding='post', maxlen=100)

    # Predict using the RNN model
    rnn_pred = model.predict(text_padded)
    rnn_label = np.argmax(rnn_pred, axis=1)

    # Example logic for categorizing text based on keywords
    for category, keywords in category_keywords.items():
        if any(keyword in text.lower() for keyword in keywords):
            return category, f"Text matched with category: {category}"

    return label_encoder.inverse_transform(rnn_label)[0], "Category prediction based on RNN model"

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Check if 'text' key is in the received JSON
        if 'text' not in data:
            return jsonify({'error': "Missing 'text' field in the request body"}), 400

        # Extract the text value from the JSON body
        text = data['text']
        
        # Call combined prediction logic
        predicted_label, message = combined_predict(text, model, tokenizer, label_encoder, category_keywords)

        # Return the prediction result as JSON
        return jsonify({'prediction': predicted_label, 'message': message})

    except Exception as e:
        # Return any errors encountered
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
