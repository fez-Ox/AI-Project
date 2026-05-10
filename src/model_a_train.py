import os

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    silhouette_score,
)
from sklearn.svm import LinearSVC


def load_data():
    """Load the processed binary datasets."""
    train_df = (
        pd.read_csv("data/processed/train_binary.csv")
        if os.path.exists("data/processed/train_binary.csv")
        else None
    )
    val_df = (
        pd.read_csv("data/processed/val_binary.csv")
        if os.path.exists("data/processed/val_binary.csv")
        else None
    )
    return train_df, val_df


def run_unsupervised(X_train_tfidf):
    """Run K-Means clustering as the mandatory unsupervised component."""
    print("Running Unsupervised Component (K-Means Clustering)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_train_tfidf)

    # Calculate silhouette score using a subset if the matrix is too large
    score = silhouette_score(
        X_train_tfidf, cluster_labels, sample_size=10000, random_state=42
    )
    print(f"K-Means Silhouette Score: {score:.4f}\n")


def train_and_evaluate():
    train_df, val_df = load_data()

    if train_df is None or val_df is None:
        print("Processed data not found. Please run preprocessing.py first.")
        return

    print("Initializing TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    # Fit on training data ONLY to avoid data leakage
    print("Fitting Vectorizer and Transforming Training Data...")
    X_train = vectorizer.fit_transform(train_df["combined_text"].fillna(""))
    y_train = train_df["label"]

    # Transform validation data
    print("Transforming Validation Data...")
    X_val = vectorizer.transform(val_df["combined_text"].fillna(""))
    y_val = val_df["label"]

    # Run Unsupervised Component
    run_unsupervised(X_train)

    # Train Logistic Regression
    print("Training Logistic Regression Model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    # Evaluate LR
    lr_preds = lr_model.predict(X_val)
    print("--- Logistic Regression Results ---")
    print(f"Accuracy: {accuracy_score(y_val, lr_preds):.4f}")
    print(f"Macro F1: {f1_score(y_val, lr_preds, average='macro'):.4f}")
    print(classification_report(y_val, lr_preds))

    # Train SVM
    print("Training Linear SVM Model...")
    svm_model = LinearSVC(random_state=42, dual="auto")
    svm_model.fit(X_train, y_train)

    # Evaluate SVM
    svm_preds = svm_model.predict(X_val)
    print("--- Linear SVM Results ---")
    print(f"Accuracy: {accuracy_score(y_val, svm_preds):.4f}")
    print(f"Macro F1: {f1_score(y_val, svm_preds, average='macro'):.4f}")
    print(classification_report(y_val, svm_preds))

    # Save the models and vectorizer
    os.makedirs("models/model_a", exist_ok=True)
    print("Saving Models and Vectorizer...")
    joblib.dump(vectorizer, "models/model_a/vectorizer.pkl")
    joblib.dump(lr_model, "models/model_a/lr_model.pkl")
    joblib.dump(svm_model, "models/model_a/svm_model.pkl")
    print("Training pipeline complete.")


if __name__ == "__main__":
    train_and_evaluate()
