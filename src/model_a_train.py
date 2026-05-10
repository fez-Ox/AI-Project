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


def evaluate_mcq(model_name, model, X_val, y_val):
    """
    Evaluates the model exactly how it will be used in inference:
    By grouping the 4 options for each question and picking the one with the highest score (Argmax).
    """
    import numpy as np

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_val)[:, 1]
    else:
        scores = model.decision_function(X_val)

    # Reshape the 1D arrays into 2D arrays where each row represents 1 question and its 4 options
    # The data is inherently ordered A, B, C, D in preprocessing.py
    try:
        scores_reshaped = scores.reshape(-1, 4)
        y_val_reshaped = y_val.values.reshape(-1, 4)

        # Get the index (0, 1, 2, or 3) of the option the model predicted as Correct
        preds_idx = np.argmax(scores_reshaped, axis=1)
        # Get the index of the option that is actually Correct
        true_idx = np.argmax(y_val_reshaped, axis=1)

        acc = accuracy_score(true_idx, preds_idx)
        print(f"--- {model_name} Results ---")
        print(f"Real MCQ Accuracy (Argmax): {acc:.4f}\n")
    except ValueError:
        print(
            f"Warning: Could not perform Argmax evaluation because validation dataset length ({len(scores)}) is not a multiple of 4."
        )


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
    from scipy.sparse import csr_matrix, hstack

    from src.utils import get_overlap_ratio

    print("Fitting Vectorizer and Transforming Training Data...")
    X_train_tfidf = vectorizer.fit_transform(train_df["combined_text"].fillna(""))
    y_train = train_df["label"]

    # Compute custom features for training
    print("Computing Word Overlap Features for Training Data...")
    train_overlap = train_df.apply(
        lambda r: get_overlap_ratio(r["article"], r["option"]), axis=1
    ).values.reshape(-1, 1)
    X_train = hstack([X_train_tfidf, csr_matrix(train_overlap)])

    # Transform validation data
    print("Transforming Validation Data...")
    X_val_tfidf = vectorizer.transform(val_df["combined_text"].fillna(""))
    y_val = val_df["label"]

    # Compute custom features for validation
    val_overlap = val_df.apply(
        lambda r: get_overlap_ratio(r["article"], r["option"]), axis=1
    ).values.reshape(-1, 1)
    X_val = hstack([X_val_tfidf, csr_matrix(val_overlap)])

    # Run Unsupervised Component
    run_unsupervised(X_train)

    # Train Logistic Regression
    print("Training Logistic Regression Model...")
    lr_model = LogisticRegression(
        C=5.0, max_iter=1000, random_state=42, class_weight="balanced"
    )
    lr_model.fit(X_train, y_train)

    # Evaluate LR
    lr_preds = lr_model.predict(X_val)
    print("--- Logistic Regression (Binary Threshold) ---")
    print(f"Accuracy: {accuracy_score(y_val, lr_preds):.4f}")
    print(f"Macro F1: {f1_score(y_val, lr_preds, average='macro'):.4f}")
    print(classification_report(y_val, lr_preds, zero_division=0))
    evaluate_mcq("Logistic Regression", lr_model, X_val, y_val)

    # Train SVM
    print("Training Linear SVM Model...")
    svm_model = LinearSVC(C=1.0, random_state=42, dual="auto", class_weight="balanced")
    svm_model.fit(X_train, y_train)

    # Evaluate SVM
    svm_preds = svm_model.predict(X_val)
    print("--- Linear SVM (Binary Threshold) ---")
    print(f"Accuracy: {accuracy_score(y_val, svm_preds):.4f}")
    print(f"Macro F1: {f1_score(y_val, svm_preds, average='macro'):.4f}")
    print(classification_report(y_val, svm_preds, zero_division=0))
    evaluate_mcq("Linear SVM", svm_model, X_val, y_val)

    # Save the models and vectorizer
    os.makedirs("models/model_a", exist_ok=True)
    print("Saving Models and Vectorizer...")
    joblib.dump(vectorizer, "models/model_a/vectorizer.pkl")
    joblib.dump(lr_model, "models/model_a/lr_model.pkl")
    joblib.dump(svm_model, "models/model_a/svm_model.pkl")
    print("Training pipeline complete.")


if __name__ == "__main__":
    train_and_evaluate()
