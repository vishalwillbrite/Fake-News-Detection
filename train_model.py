# ============================================
# Fake News Detection System
# Model Training Script
# ============================================

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import preprocess_text

# ============================================
# Step 1 - Load Dataset
# ============================================

print("Loading datasets...")

fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# ============================================
# Step 2 - Add Labels
# ============================================

fake_df["label"] = 0
true_df["label"] = 1

# ============================================
# Step 3 - Merge Dataset
# ============================================

news_df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle dataset
news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total News Articles : {len(news_df)}")

# ============================================
# Step 4 - Preprocess Text
# ============================================

print("Cleaning news articles...")

news_df["text"] = news_df["text"].apply(preprocess_text)

# ============================================
# Step 5 - Feature Extraction
# ============================================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(news_df["text"])

y = news_df["label"]

# ============================================
# Step 6 - Split Dataset
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# Step 7 - Train Model
# ============================================

print("Training Logistic Regression model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ============================================
# Step 8 - Prediction
# ============================================

y_pred = model.predict(X_test)

# ============================================
# Step 9 - Evaluation
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print(f"Accuracy : {accuracy * 100:.2f}%")
print("===================================\n")

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ============================================
# Step 10 - Save Model
# ============================================

joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel saved successfully.")
print("Vectorizer saved successfully.")