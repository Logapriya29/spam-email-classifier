import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Spam Classifier Dashboard", layout="wide")
st.title("SMS Spam Classifier Dashboard")
st.markdown("### Model Performance and Live Prediction")

# --- DATA LOADING AND MODEL TRAINING ---
@st.cache_data
def load_and_train_model():
    # Load the dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Prepare data for training
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data
    cv = CountVectorizer(stop_words='english')
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)

    # Train the Multinomial Naive Bayes model
    classifier = MultinomialNB()
    classifier.fit(X_train_cv, y_train)

    # Make predictions and get metrics
    y_pred = classifier.predict(X_test_cv)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return classifier, cv, acc, prec, rec, cm, y_test, y_pred

# Load the trained model and data
classifier, cv, accuracy, precision, recall, cm, y_test, y_pred = load_and_train_model()

# --- LIVE PREDICTION FEATURE ---
st.header("Predict and Detect Fake Spam")
st.markdown("Enter a message below to see the prediction and confirm if it's correct.")

# Create the text input field
user_input = st.text_area("Enter your message:", height=150, key="user_input")

# Create the prediction button
if st.button("Predict"):
    if user_input:
        # Preprocess the user's input
        user_input_cv = cv.transform([user_input])
        # Make the prediction
        prediction = classifier.predict(user_input_cv)
        
        # Store the prediction in session state to persist after a re-run
        st.session_state['last_prediction'] = prediction[0]
        st.session_state['input_text'] = user_input

        # Display the initial prediction
        if st.session_state['last_prediction'] == 1:
            st.error("🚨 This message is predicted as **SPAM**.")
        else:
            st.success("✅ This message is predicted as **NOT SPAM (Ham)**.")

    else:
        st.warning("Please enter some text to predict.")

# --- FEEDBACK SECTION (Only appears after a prediction is made) ---
if 'last_prediction' in st.session_state:
    st.markdown("---")
    st.subheader("Was the Prediction Correct?")
    
    # Get the user's feedback
    true_label_options = {0: "Actually NOT SPAM (Ham)", 1: "Actually SPAM"}
    true_label = st.radio(
        "Please tell us what the message actually is:",
        options=list(true_label_options.keys()),
        format_func=lambda x: true_label_options[x],
        key="true_label_radio"
    )

    # Button to confirm feedback
    if st.button("Confirm"):
        st.write("Thank you for your feedback!")
        predicted_label = st.session_state['last_prediction']

        if predicted_label == true_label:
            if predicted_label == 1:
                st.info("✅ **True Positive:** The model correctly predicted SPAM!")
            else:
                st.info("✅ **True Negative:** The model correctly predicted NOT SPAM.")
        else:
            if predicted_label == 1 and true_label == 0:
                st.error("❌ **False Positive (Fake Spam):** The model incorrectly predicted SPAM!")
            else:
                st.error("❌ **False Negative:** The model incorrectly predicted NOT SPAM!")

# --- DISPLAYING MODEL PERFORMANCE ---
st.header("Model Performance Metrics")

# Create three columns for the metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{accuracy*100:.2f}%")
with col2:
    st.metric("Precision", f"{precision*100:.2f}%")
with col3:
    st.metric("Recall", f"{recall*100:.2f}%")

st.markdown("---")

# --- VISUALIZING THE CONFUSION MATRIX ---
st.header("Confusion Matrix")
st.markdown("The confusion matrix helps us understand the model's performance in more detail.")

# Get counts from the confusion matrix
tn, fp, fn, tp = cm.ravel()

cm_df = pd.DataFrame(cm, 
                    index=['Actual Ham', 'Actual Spam'], 
                    columns=['Predicted Ham', 'Predicted Spam'])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix for SMS Spam Classification')
st.pyplot(fig)

# Explanation of confusion matrix values
st.subheader("Confusion Matrix Breakdown")
st.markdown(f"""
-   **True Ham ({tn}):** The number of ham messages correctly identified as ham.
-   **True Spam ({tp}):** The number of spam messages correctly identified as spam.
-   **Fake Ham (False Negatives) ({fn}):** The number of actual spam messages incorrectly classified as ham.
-   **Fake Spam (False Positives) ({fp}):** The number of actual ham messages incorrectly classified as spam.
""")
