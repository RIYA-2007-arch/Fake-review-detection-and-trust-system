import pandas as pd
import pickle
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from scipy.sparse import hstack

# ================= LOAD DATA =================
df = pd.read_csv("reviews.csv")

# ================= CLEAN TEXT =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return " ".join(text.split())

df['clean'] = df['text_'].apply(clean_text)

# ================= LABEL FIX =================
# 0 = Fake (CG), 1 = Genuine (OR)
df['label'] = df['label'].map({'CG': 0, 'OR': 1})

df = df.dropna()
df = df.drop_duplicates()

# ================= BALANCE DATASET =================
fake = df[df['label'] == 0]
genuine = df[df['label'] == 1]

min_size = min(len(fake), len(genuine))

fake = fake.sample(min_size, random_state=42)
genuine = genuine.sample(min_size, random_state=42)

df = pd.concat([fake, genuine]).sample(frac=1, random_state=42)

# ================= EXTRA FEATURES =================
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

extra = np.array([extra_features(t) for t in df['text_']])

# ================= SPLIT =================
X_train_text, X_test_text, X_train_ex, X_test_ex, y_train, y_test = train_test_split(
    df['clean'],
    extra,
    df['label'],
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

# ================= TF-IDF =================
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=22000,
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# ================= SCALE FEATURES =================
scaler = StandardScaler()
X_train_ex = scaler.fit_transform(X_train_ex)
X_test_ex = scaler.transform(X_test_ex)

# ================= COMBINE =================
X_train_final = hstack([X_train_vec, X_train_ex])
X_test_final = hstack([X_test_vec, X_test_ex])

# ================= MODEL =================
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=80,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train_final, y_train)

# ================= EVALUATION =================
y_pred = model.predict(X_test_final)
accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Accuracy:", round(accuracy * 100, 2), "%")

# ================= SAVE =================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model, Vectorizer, Scaler saved successfully")