# ============================================
# Fake News Detection System
# Phase 1 - Data Loading & Exploratory Analysis
# ============================================

# Import Libraries
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
nltk.download("stopwords")
stemmer = PorterStemmer()

# --------------------------------------------
# Step 1 - Load Dataset
# --------------------------------------------

fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# --------------------------------------------
# Step 2 - Display Basic Information
# --------------------------------------------

print("=" * 60)
print("FAKE NEWS DATASET")
print("=" * 60)

print(fake_df.head())
print("\nShape:", fake_df.shape)
print("\nColumns:")
print(fake_df.columns)

print("\nData Types:")
print(fake_df.dtypes)

print("\nMissing Values:")
print(fake_df.isnull().sum())

print("\n\n")

print("=" * 60)
print("REAL NEWS DATASET")
print("=" * 60)

print(true_df.head())
print("\nShape:", true_df.shape)

print("\nColumns:")
print(true_df.columns)

print("\nData Types:")
print(true_df.dtypes)

print("\nMissing Values:")
print(true_df.isnull().sum())

# --------------------------------------------
# Step 3 - Add Labels
# --------------------------------------------

fake_df["label"] = 0
true_df["label"] = 1

# --------------------------------------------
# Step 4 - Merge Dataset
# --------------------------------------------

news_df = pd.concat([fake_df, true_df], ignore_index=True)

# --------------------------------------------
# Step 5 - Shuffle Dataset
# --------------------------------------------

news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)

# --------------------------------------------
# Step 6 - Combined Dataset Information
# --------------------------------------------

print("\nCombined Dataset Shape:", news_df.shape)

print("\nFirst Five Rows:")
print(news_df.head())

print("\nLabel Counts:")
print(news_df["label"].value_counts())

# --------------------------------------------
# Step 7 - Label Distribution
# --------------------------------------------

news_df["label"].value_counts().plot(
    kind="bar",
    figsize=(6,4)
)

plt.title("Fake vs Real News")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks([0,1],["Fake","Real"],rotation=0)

plt.show()

# --------------------------------------------
# Step 8 - Dataset Information
# --------------------------------------------

print(news_df.info())

print(news_df.describe(include="all"))
# ==========================================
# Step 11 - Text Preprocessing Function
# ==========================================

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Split into words
    words = text.split()

    # Remove stopwords and apply stemming
    cleaned_words = []

    for word in words:

        if word not in stopwords.words("english"):

            cleaned_words.append(stemmer.stem(word))

    # Join words back into a sentence
    return " ".join(cleaned_words)
  
    # ==========================================
news_df["text"] = news_df["text"].apply(preprocess_text)

print(news_df["text"].head())
# Step 12 - TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(news_df["text"])
print("Feature Matrix Shape:", X.shape)
y = news_df["label"]
print("Target Shape:", y.shape)
print(vectorizer.get_feature_names_out()[:20])
# ==========================================
# Step 13 - Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# ==========================================
# Step 14 - Create Logistic Regression Model
# ==========================================

model = LogisticRegression(max_iter=1000)
# ==========================================
# Step 15 - Train Model
# ==========================================

model.fit(X_train, y_train)
# ==========================================
# Step 16 - Prediction
# ==========================================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# ==========================================
# Step 17 - Save Model
# ==========================================

joblib.dump(model, "models/fake_news_model.pkl")

print("Model saved successfully!")
# ==========================================
# Step 18 - Save TF-IDF Vectorizer
# ==========================================

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Vectorizer saved successfully!")
