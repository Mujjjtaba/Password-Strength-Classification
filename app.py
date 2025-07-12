import streamlit as st
import joblib
import random
import string
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ✅ Define char_tokenizer to match what was used during training
def char_tokenizer(text):
    return list(text)

# ✅ Load Models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl")
}

# ✅ Load TF-IDF Vectorizer
tfidf = joblib.load("models/tfidf.pkl")

# ✅ Password Strength Prediction
def predict_password_strength(password, model):
    password_features = tfidf.transform([password])
    prediction = model.predict(password_features)[0]
    return ["Weak", "Moderate", "Strong"][prediction]

# ✅ Password Generator
def generate_strong_password(length=12):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

# ✅ Cached Dataset Load
@st.cache_data
def load_eval_data():
    df = pd.read_csv("data/data.csv")
    df['password'] = df['password'].fillna("")
    X = df['password']
    y = df['strength']
    X_tfidf = tfidf.transform(X)
    return train_test_split(X_tfidf, y, test_size=0.1, random_state=42)

# ✅ Confusion Matrix Evaluation
def show_model_evaluation():
    st.title("📊 Model Evaluation - Confusion Matrices")
    X_train, X_test, y_train, y_test = load_eval_data()
    for name, model in models.items():
        st.markdown(f"---\n### {name} - Confusion Matrix")
        with st.spinner(f"Evaluating {name}..."):
            X_sample, y_sample = resample(X_test, y_test, n_samples=3000, random_state=42)
            y_pred = model.predict(X_sample)
            cm = confusion_matrix(y_sample, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Weak", "Moderate", "Strong"],
                        yticklabels=["Weak", "Moderate", "Strong"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

# ✅ Training Time + Accuracy Comparison
def show_model_comparison():
    st.title("📈 Model Comparison: Training Time & Accuracy")

    try:
        with open("models/metrics.json", "r") as f:
            metrics = json.load(f)

        df = pd.DataFrame(metrics).T.reset_index()
        df.rename(columns={
            "index": "Model",
            "training_time_sec": "Training Time (sec)",
            "accuracy": "Accuracy (%)"
        }, inplace=True)

        st.subheader("📊 Comparison Table")
        st.dataframe(df)

        st.subheader("⏱ Training Time Comparison")
        fig1, ax1 = plt.subplots()
        sns.barplot(data=df, x="Model", y="Training Time (sec)", ax=ax1)
        st.pyplot(fig1)

        st.subheader("✅ Accuracy Comparison")
        fig2, ax2 = plt.subplots()
        sns.barplot(data=df, x="Model", y="Accuracy (%)", ax=ax2)
        st.pyplot(fig2)

    except FileNotFoundError:
        st.error("❌ metrics.json not found. Please re-run `train_and_save_models.py`.")

# ✅ Streamlit App
def main():
    st.title("🔐 Password Strength Classifier ML App")
    st.subheader("Built with Streamlit")

    activities = [
        "Classify Password",
        "Generate Password",
        "About",
        "Model Evaluation",
        "Model Comparison"
    ]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Classify Password":
        st.subheader("Classify Your Password")
        password = st.text_input("Enter Password:")
        model_choice = st.selectbox("Select Model", list(models.keys()))
        if st.button("Check Password Strength"):
            result = predict_password_strength(password, models[model_choice])
            st.success(f"Password Strength: {result}")

    elif choice == "Generate Password":
        st.subheader("Generate Strong Password")
        length = st.slider("Select Length", 8, 24, 12)
        if st.button("Generate Password"):
            password = generate_strong_password(length)
            st.info(f"Generated Password: {password}")

    elif choice == "About":
        st.subheader("About App")
        st.info("This app classifies password strength using ML models like Logistic Regression, XGBoost, and Naive Bayes. It also generates strong random passwords.")

    elif choice == "Model Evaluation":
        show_model_evaluation()

    elif choice == "Model Comparison":
        show_model_comparison()

# ✅ Run App
if __name__ == '__main__':
    main()
