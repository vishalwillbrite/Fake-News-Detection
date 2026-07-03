# ==============================
# Fake News Detection System
# Step 1 - Import Libraries
# ==============================

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
