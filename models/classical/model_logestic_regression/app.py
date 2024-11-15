import streamlit as st
import pickle
import string
import pandas as pd

# Load the pre-trained model, vectorizer, and encoders
with open('crime_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('category_encoder.pkl', 'rb') as f:
    category_encoder = pickle.load(f)

with open('sub_category_encoder.pkl', 'rb') as f:
    sub_category_encoder = pickle.load(f)

# Function to preprocess user input by removing punctuation
def preprocess_text(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Streamlit app UI
st.title("Crime Classification App")
st.write("Enter crime details below, and the app will classify the crime into a category and sub-category.")

# Get user input
user_input = st.text_area("Enter crime description", "")

# Button to classify
if st.button("Classify"):
    if user_input.strip() == "":
        st.write("Please enter a valid crime description.")
    else:
        # Preprocess input text
        processed_input = preprocess_text(user_input)
        
        # Predict using the model
        prediction = model.predict([processed_input])
        
        # Decode the labels back to their original categories
        category = category_encoder.inverse_transform([prediction[0][0]])[0]
        sub_category = sub_category_encoder.inverse_transform([prediction[0][1]])[0]
        
        # Display the predictions
        st.write("### Predicted Category:", category)
        st.write("### Predicted Sub-Category:", sub_category)
