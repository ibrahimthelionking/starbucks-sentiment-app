# ☕ Starbucks Review Sentiment Analysis

This repository contains an end-to-end Natural Language Processing (NLP) pipeline that classifies customer reviews as Positive or Negative, along with a Streamlit web app for real-time inference.

## The Motivation
Customer feedback is the lifeblood of the service industry. I built this project to demonstrate how raw, messy text data can be cleaned, vectorized, and used to train accurate predictive models for automated sentiment monitoring.

## The Approach
- **EDA & Cleaning:** Handled missing ratings, duplicates, and placeholder text. Implemented text preprocessing including lowercasing, regex noise removal, and lemmatization. Crucially, negation words (e.g., "not", "never") were preserved to maintain semantic meaning.
- **Labeling:** Converted 1-5 star ratings into a binary classification (Positive/Negative) and dropped neutral 3-star reviews to improve model confidence.
- **Modeling:** Tested Naive Bayes, Logistic Regression, and Support Vector Machines (SVM) using TF-IDF vectorization. The SVM pipeline achieved the best performance.
- **Deployment:** Saved the full pipeline (vectorizer + model) using `joblib` and built a Streamlit app for live predictions.

## Run it Locally
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

*Note: The trained model file (`sentiment_model.joblib`) is included in this repository.*
