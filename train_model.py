# ==============================
# Fake News Detection System
# Step 1 - Import Libraries
# ==============================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==============================
# Step 2 - Load Dataset
# ==============================

fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# ==============================
# Step 3 - Display Basic Information
# ==============================

print("=" * 60)
print("FAKE NEWS DATASET")
print("=" * 60)

print(fake_df.head())

print("\nShape :", fake_df.shape)

print("\nColumns :")
print(fake_df.columns)

print("\nData Types :")
print(fake_df.dtypes)

print("\nMissing Values :")
print(fake_df.isnull().sum())


print("\n\n")

print("=" * 60)
print("REAL NEWS DATASET")
print("=" * 60)

print(true_df.head())

print("\nShape :", true_df.shape)

print("\nColumns :")
print(true_df.columns)

print("\nData Types :")
print(true_df.dtypes)

print("\nMissing Values :")
print(true_df.isnull().sum())
# ==============================
# Step 4 - Add Labels
# ==============================

fake_df["label"] = 0      # Fake News
true_df["label"] = 1      # Real News
# ==============================
# Step 5 - Merge Datasets
# ==============================

news_df = pd.concat([fake_df, true_df], ignore_index=True)
# ==============================
# Step 6 - Shuffle Dataset
# ==============================

news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)
# ==============================
# Step 7 - Combined Dataset Info
# ==============================

print("\nCombined Dataset Shape:", news_df.shape)

print("\nFirst 5 Rows:")
print(news_df.head())

print("\nLabel Counts:")
print(news_df["label"].value_counts())
# ==============================
# Step 8 - Count Labels
# ==============================

print(news_df["label"].value_counts())
# ==============================
# Step 9 - Plot Label Distribution
# ==============================

news_df["label"].value_counts().plot(
    kind="bar",
    figsize=(6,4)
)

plt.title("Fake vs Real News")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks([0,1], ["Fake","Real"], rotation=0)

plt.show()
# ==============================
# Step 10 - Dataset Information
# ==============================

print(news_df.info())
print(news_df.describe(include="all"))