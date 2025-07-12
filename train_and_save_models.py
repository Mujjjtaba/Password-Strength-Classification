import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import time
import json
 
def char_tokenizer(text):
    return list(text)
 
print("📦 Loading dataset...")
df = pd.read_csv('data/data.csv')
X = df['password'].fillna("")
y = df['strength']

print("🧠 Vectorizing with TF-IDF...")
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

print("🚀 Starting model training...")
for name, model in models.items():
    try:
        print(f"\n🔧 Training model: {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()

        joblib.dump(model, f"models/{name}.pkl")

        accuracy = accuracy_score(y_test, model.predict(X_test))
        metrics[name] = {
            "training_time_sec": round(end - start, 2),
            "accuracy": round(accuracy * 100, 2)
        }

        print(f"✅ {name} saved successfully. Accuracy: {metrics[name]['accuracy']}%, Time: {metrics[name]['training_time_sec']}s")

    except Exception as e:
        print(f"❌ ERROR training {name}: {e}")

joblib.dump(tfidf, "models/tfidf.pkl")
print("\n📁 TF-IDF vectorizer saved.")

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("📊 Training metrics saved.")
