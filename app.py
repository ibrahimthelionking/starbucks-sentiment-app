import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data (Streamlit needs this on startup)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

st.set_page_config(page_title="Starbucks Sentiment", page_icon="☕")
st.title("☕ Starbucks Review Sentiment Analyzer")
st.write("Type a review below to see if the AI thinks it is Positive or Negative.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('sentiment_model.joblib')

model = load_model()

# The exact clean_text function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Uncomment the line below ONLY if you kept negation words in your notebook
# stop_words = stop_words - {'not', 'no', 'never'}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# UI
user_input = st.text_area("Enter your review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)
        
        # Predict
        prediction = model.predict([cleaned_input])[0]
        
        st.divider()
        st.subheader("Result")
        
        # The dataset used 5 for Positive and 1 for Negative
        if prediction == 5 or prediction == 1 or prediction == 'Positive':
            st.success("✅ **Positive Review!** The customer seems happy.")
        else:
            st.error("❌ **Negative Review.** The customer seems unhappy.")
