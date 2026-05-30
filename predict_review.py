import pickle
import numpy as np
from scipy.sparse import hstack
import re

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text (same as training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = " ".join(text.split())
    return text

# Extra features (same as training)
def extra_features(text):
    words = text.split()
    return [
        len(words),
        len(text),
        text.count("!"),
        text.count("?"),
        len(set(words)),
        int(text.isupper())
    ]

# Input
user_review = input("Enter a review: ")

# Preprocess
cleaned = clean_text(user_review)

vec = vectorizer.transform([cleaned])
extra = np.array([extra_features(cleaned)])

final_input = hstack([vec, extra])

# Predict
prediction = model.predict(final_input)

# Output
if prediction[0] == 1:
    print("Genuine Review")
else:
    print("Fake Review")