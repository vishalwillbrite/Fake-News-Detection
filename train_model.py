# ============================================
# Fake News Detection System
# Phase 1 - Data Loading & Exploratory Analysis
# ============================================


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
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
    news_df["text"] = news_df["text"].apply(preprocess_text)
    print(news_df["text"].head())