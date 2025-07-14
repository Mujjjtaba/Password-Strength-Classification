import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import joblib
import os
import time
import json

def char_tokenizer(text):
    return list(text)

print("📦 Loading dataset...")
df = pd.read_csv('data/data.csv')
df['password'] = df['password'].fillna("")

print("\n🔵 Class distribution:")
print(df['strength'].value_counts(normalize=True))

print("\n🔁 Balancing the dataset...")
df_major = df[df['strength'] == 1]
df_minor_0 = df[df['strength'] == 0]
df_minor_2 = df[df['strength'] == 2]

df_0_upsampled = resample(df_minor_0, replace=True, n_samples=len(df_major), random_state=42)
df_2_upsampled = resample(df_minor_2, replace=True, n_samples=len(df_major), random_state=42)

df_balanced = pd.concat([df_major, df_0_upsampled, df_2_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X = df_balanced['password']
y = df_balanced['strength']

print("\n🧠 Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(tokenizer=char_tokenizer, token_pattern=None)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "naive_bayes": MultinomialNB()
}

os.makedirs("models", exist_ok=True)
metrics = {}

print("\n🚀 Starting model training...")
for name, model in models.items():
    try:
        print(f"\n🔧 Training model: {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()

        joblib.dump(model, f"models/{name}.pkl")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        metrics[name] = {
            "training_time_sec": round(end - start, 2),
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2)
        }

        print(f"✅ {name} saved. Accuracy: {metrics[name]['accuracy']}%, Time: {metrics[name]['training_time_sec']}s")

    except Exception as e:
        print(f"❌ ERROR training {name}: {e}")

# Save vectorizer
joblib.dump(tfidf, "models/tfidf.pkl")
print("\n📁 TF-IDF vectorizer saved.")

# Save all metrics to JSON
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("📊 Training metrics (with extended metrics) saved.")
