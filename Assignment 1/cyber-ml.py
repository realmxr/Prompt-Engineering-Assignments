import pandas as pd
import matplotlib.pyplot as plt

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

# ---------------- Load ----------------
df = pd.read_csv("spam.csv", encoding="latin-1")

# Use only the two relevant columns
df = df.rename(columns={"Category": "label", "Message": "message"})[["label", "message"]]

# Drop NAs and normalize
df = df.dropna()
df["message"] = df["message"].astype(str)
df["label"] = df["label"].str.strip().str.lower()

print("Label counts:\n", df["label"].value_counts())

# ---------------- Split ----------------
X = df["message"]
y = df["label"]

# Stratified split to maintain label distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------- Logistic Regression ----------------
log_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),  # Using TF-IDF for text features
    ("clf", LogisticRegression(max_iter=5000))  # Increased max_iter to ensure convergence
])

print("\nTraining Logistic Regression...")
log_pipe.fit(X_train, y_train)
pred_log = log_pipe.predict(X_test)

print("Logistic Regression Results:")
print("Accuracy:", f"{accuracy_score(y_test, pred_log):.4f}")
print(classification_report(y_test, pred_log))

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(log_pipe, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix — Logistic Regression")
plt.show()

# ---------------- Multinomial Naive Bayes ----------------
nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB())
])

print("\nTraining Multinomial Naive Bayes...")
nb_pipe.fit(X_train, y_train)
pred_nb = nb_pipe.predict(X_test)

print("Naive Bayes Results:")
print("Accuracy:", f"{accuracy_score(y_test, pred_nb):.4f}")
print(classification_report(y_test, pred_nb))

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(nb_pipe, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix — Multinomial Naive Bayes")
plt.show()

print("\nMachine learning classification completed.")
