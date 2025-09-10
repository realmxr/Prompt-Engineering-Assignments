import pandas as pd
import matplotlib.pyplot as plt

# scikit-learn tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay
)

# Set random state for reproducibility
RANDOM_STATE = 42

print("Starting SMS Spam Classification...")

# Load the dataset into a DataFrame
df = pd.read_csv("spam.csv", encoding="latin-1")

# Use only the two relevant columns
df = df.rename(columns={"Category": "label", "Message": "message"})[["label", "message"]]

# Drop NAs and normalize labels
df = df.dropna()
df["message"] = df["message"].astype(str)
df["label"] = df["label"].str.strip().str.lower()

# Display basic information about the dataset
print("Label counts:\n", df["label"].value_counts())

# Split the dataset into features and target (feature: message, target: spam/ham)
X = df["message"]
y = df["label"]

# Split the dataset into training and testing sets (80% train, 20% test)
# Stratifying ensures balanced class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

log_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),  # Convert text into numerical vectors
    ("clf", LogisticRegression(max_iter=5000))  # Logistic Regression classifier
])

print("\nTraining Logistic Regression...")
log_pipe.fit(X_train, y_train)  # Train the model
pred_log = log_pipe.predict(X_test) # Make predictions on the test set

print("Logistic Regression Results:")
print("Accuracy:", f"{accuracy_score(y_test, pred_log):.4f}")
print(classification_report(y_test, pred_log))

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(log_pipe, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix — Logistic Regression")
plt.show()

nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")), # Convert text into numerical vectors
    ("clf", MultinomialNB()) # Multinomial Naive Bayes classifier
])

print("\nTraining Multinomial Naive Bayes...")
nb_pipe.fit(X_train, y_train)   # Train the model
pred_nb = nb_pipe.predict(X_test)   # Make predictions on the test set

print("Naive Bayes Results:")
print("Accuracy:", f"{accuracy_score(y_test, pred_nb):.4f}")
print(classification_report(y_test, pred_nb))

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(nb_pipe, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix — Multinomial Naive Bayes")
plt.show()

print("\nMachine learning classification completed.")
