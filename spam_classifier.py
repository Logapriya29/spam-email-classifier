import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and Prepare Data (Using a simplified dummy dataset) ---
# In a real project, you would load a CSV file like 'spam.csv'
data = {
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'ham'],
    'text': [
        "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got no monty... 5",
        "WINNER! U have been selected to receive a £900 prize! To claim, call 0906...",
        "Ok lar... Joking wif u oni...",
        "URGENT! Your mobile number has won a £5000 prize. Call now on 0800...",
        "Nah I don't think he goes to usf, he lives around here though",
        "Congratulations! You've won a FREE trip! Click the link now.",
        "I'm back home now, do you want to meet up later?",
        "Free entry in 2 a wkly comp to win FA Cup final tickets 21st May 2005. Text FA to 87121",
        "Had your mobile 11 months or more? U R entitled to a FREE H&M gift!",
        "Rofl. Its true to its name."
    ]
}
df = pd.DataFrame(data)

# Map labels to 0 (Ham) and 1 (Spam)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_num'], test_size=0.3, random_state=42
)

# --- 2. Feature Extraction and Model Training (TF-IDF and Naive Bayes) ---
# Convert text to numerical feature vectors
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a classifier (Naive Bayes is a common choice for text)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = model.predict(X_test_vectorized)

# --- 3. Calculate and Display Confusion Metrics (Dashboard Data) ---
# Generate the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix components (order: TN, FP, FN, TP)
TN, FP, FN, TP = cm.ravel()

# Create a clean DataFrame for dashboard visualization
dashboard_data = {
    'Category': ['True Spam (TP)', 'Fake Spam (FP)', 'True Ham (TN)', 'Fake Ham (FN)'],
    'Description': ['Correctly classified Spam', 'Ham incorrectly classified as Spam', 'Correctly classified Ham', 'Spam incorrectly classified as Ham'],
    'Count': [TP, FP, TN, FN],
    'Correct': ['Yes', 'No', 'Yes', 'No']
}
dashboard_df = pd.DataFrame(dashboard_data)

# Print the results
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("\n--- Confusion Matrix Data for Dashboard ---")
print(dashboard_df)

# --- 4. Simple Visualization (Can be part of the dashboard) ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Ham', 'Predicted Spam'],
            yticklabels=['Actual Ham', 'Actual Spam'])
plt.title('Spam Classifier Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()