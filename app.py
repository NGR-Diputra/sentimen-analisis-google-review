from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np
import json
from pathlib import Path

from utils import (
    case_folding,
    load_abbreviation_file,
    normalize_text,
    remove_custom_stopwords,
    stemming_text
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Get the directory where app.py is located
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# RUNNING THE FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

model_path = BASE_DIR / "model.h5"

model = load_model(model_path)


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer_config_path = BASE_DIR / "tokenizer_config.json"

with open(tokenizer_config_path, "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

    max_words = config["num_words"]

    tokenizer = Tokenizer(
        num_words=max_words,
        oov_token="<OOV>"
    )

    tokenizer.word_index = json.loads(config["word_index"])

    tokenizer.index_word = {
        str(i): word
        for word, i in tokenizer.word_index.items()
    }


# ============================================================
# LOAD CUSTOM STOPWORDS PATH
# ============================================================

custom_stopwords_file = (
    BASE_DIR / "model" / "more_stopwords.txt"
)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    result = perform_sentiment_analysis(text)

    return render_template(
        "index.html",
        result=result,
        text=text
    )


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    # 1. Case folding
    text = case_folding(text)

    # 2. Normalize abbreviations
    text = normalize_text(text)

    # 3. Remove custom stopwords
    text = remove_custom_stopwords(
        text,
        custom_stopwords_file
    )

    # 4. Stemming
    text = stemming_text(text)

    return text


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def perform_sentiment_analysis(text):

    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    print("Original text:", text)
    print("Preprocessed text:", preprocessed_text)

    # Tokenize and pad the sequence
    max_length = 100

    text_seq = tokenizer.texts_to_sequences(
        [preprocessed_text]
    )

    text_padded = pad_sequences(
        text_seq,
        maxlen=max_length,
        padding="post"
    )

    # Make prediction
    prediction = model.predict(text_padded)

    # Get the class with the highest probability
    max_prob_index = np.argmax(prediction)

    # Convert prediction to sentiment
    sentiment = (
        "Positive"
        if max_prob_index == 2
        else "Negative"
    )

    return sentiment


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)