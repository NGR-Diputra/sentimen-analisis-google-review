# Indonesian Google Review Sentiment Analysis

This project is a simple sentiment analysis application for **Indonesian Google Reviews**. It uses a trained machine learning model to classify a review as either **Positive** or **Negative**.

The project includes the dataset, trained model, text preprocessing tools, model training notebook, and a Flask web application that allows users to enter a review and receive a sentiment prediction.

## About the Project

The goal of this project is to analyze the sentiment expressed in Indonesian-language Google Reviews. Given a review as input, the application processes the text and uses a trained deep learning model to predict whether the review expresses a positive or negative sentiment.

This project can also serve as an example of how **Natural Language Processing (NLP)** and machine learning can be applied to customer reviews.

## Dataset

The model is developed using a dataset of Google Reviews stored in:

`model/src/dataset.xlsx`

The dataset contains review data used for developing the sentiment analysis model.

The project also includes supporting files for Indonesian text preprocessing, including:

- Abbreviation normalization
- Indonesian stopwords
- Custom stopwords
- Indonesian stemming

## How It Works

The overall process is:

```text
Review / Text Input
       ↓
Text Preprocessing
       ↓
Tokenization
       ↓
Sequence Padding
       ↓
Trained Machine Learning Model
       ↓
Sentiment Prediction
       ↓
Positive / Negative
