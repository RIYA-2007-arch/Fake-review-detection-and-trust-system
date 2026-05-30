import pickle
import numpy as np
from scipy.sparse import hstack
import re

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return " ".join(text.split())

def extra_features(text):
    words = text.split()
    diversity = len(set(words)) / (len(words) + 1)
    return [
        len(words),
        len(text),
        text.count("!"),
        text.count("?"),
        diversity,
        int(text.isupper())
    ]

while True:
    text = input("Enter review: ")

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    extra = np.array([extra_features(text)])
    extra_scaled = scaler.transform(extra)

    final = hstack([vec, extra_scaled])

    pred = model.predict(final)[0]
    prob = model.predict_proba(final)[0]

    print("Prediction:", pred)
    print("Prob:", prob)