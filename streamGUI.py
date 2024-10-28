# import the needed libraries
import re
import json 
import string
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# load the model
MODEL_PATH = 'C:\\project\\best_model.keras'
model = load_model(MODEL_PATH) 

# load the tokenizer
TOKENIZER_PATH = 'C:\\project\\tokenizer.json'
with open(TOKENIZER_PATH) as json_file:
    tokenizer_json = json.load(json_file)
tokenizer = tokenizer_from_json(tokenizer_json)

# text preprocessing 
def preprocess(sentence):
    ret = sentence.lower()
    ret = ret.translate(str.maketrans('', '', string.punctuation))
    NON_ASCII = re.compile(r'[^A-Za-z0-9\s]')
    ret = NON_ASCII.sub(r'', ret)
    return ret.strip()

# perdiction
def predict_sentence(sentence):
    MAX_LENGTH = 256
    sentence = preprocess(sentence)
    sentence = [sentence]
    tokenized_sentence = tokenizer.texts_to_sequences(sentence)
    padded_sentence = pad_sequences(tokenized_sentence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    pred = model.predict(padded_sentence)[0][0]
    return "Positive" if pred > 0.5 else "Negative"

# main function 
def main():
    # Set up the title and header with custom markdown styling
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Sentiment Analysis App 💬</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Analyze the sentiment of your thoughts in seconds! 🔍</h4>", unsafe_allow_html=True)
    st.write("---")  # Divider line for better layout

    # Input area for user text
    st.markdown("### 👉 Enter your thoughts below")
    user_input = st.text_area("What’s on your mind?", placeholder="Type something...", height=150)

    # Add an Analyze button and display results dynamically
    if st.button('Analyze Sentiment'):
        if user_input:
            # Display loading animation
            with st.spinner("Analyzing sentiment..."):
                processed_text = preprocess(user_input)
                sentiment = predict_sentence(processed_text)
            
            # Display sentiment result with dynamic color and emoji
            if sentiment == "Positive":
                st.success("😊 Your statement is **Positive**! Keep up the positivity!")
            else:
                st.warning("😟 Your statement is **Negative**. Let's aim for positivity!")


            # Feedback message
            st.markdown("---")
            st.markdown("<p style='text-align: center; font-size: 1.2em; color: #333;'>Thank you for using the Sentiment Analysis App!</p>", unsafe_allow_html=True)
        else:
            st.error("⚠️ Please enter some text to analyze.")
        
# Entry point
if __name__ == '__main__':
    main()