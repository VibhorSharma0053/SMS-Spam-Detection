import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# Streamlit UI
st.title("SMS Spam Detection App")
st.write("Enter an SMS message below to check if it is spam or not.")

# Input box for user to enter message
message = st.text_area("Enter your message here:")

if st.button("Check"):  # Check button
    if message.strip():
        processed_message = preprocess_text(message)
        transformed_message = vectorizer.transform([processed_message])
        prediction = model.predict(transformed_message)
        
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.write(f"### Result: {result}")
    else:
        st.warning("Please enter a message to check.")
