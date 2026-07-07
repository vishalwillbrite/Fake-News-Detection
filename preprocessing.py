# ============================================
# preprocessing.py
# Text Cleaning Functions
# ============================================

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (runs only once if already downloaded)
nltk.download("stopwords")

# Initialize stemmer
stemmer = PorterStemmer()

# Store stopwords in a set (faster lookup)
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Clean and preprocess news text.
    """

    # Handle missing values
    if text is None:
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Split into words
    words = text.split()

    # Remove stopwords and apply stemming
    cleaned_words = []

    for word in words:

        if word not in stop_words:
            cleaned_words.append(stemmer.stem(word))

    return " ".join(cleaned_words)