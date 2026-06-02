import streamlit as st
import pickle
import json
import random
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Load files
model = pickle.load(open("svm_intent_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

with open("intents.json") as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()

def preprocess(text):

    tokens = word_tokenize(text.lower())

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum()
    ]

    return " ".join(tokens)

def get_response(message):

    processed = preprocess(message)

    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)[0]

    tag = encoder.inverse_transform([prediction])[0]

    for intent in intents["intents"]:

        if intent["tag"] == tag:

            return random.choice(intent["responses"])

    return "I don't understand."

st.title("Kasokoso Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input(
    "Ask a question..."
)

if user_input:

    st.session_state.messages.append(
        {"role":"user","content":user_input}
    )

    response = get_response(user_input)

    st.session_state.messages.append(
        {"role":"assistant","content":response}
    )

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])