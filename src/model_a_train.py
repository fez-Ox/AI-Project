import os
import sys

# Add the project root to the Python path to allow absolute imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.svm import LinearSVC

from src.utils import get_overlap_ratio


def load_data():
    train_df = (
        pd.read_csv("data/processed/train_binary.csv")
        if os.path.exists("data/processed/train_binary.csv")
        else None
    )
    return train_df


def run_unsupervised(X_train_tfidf):
    print("Running Unsupervised Component (K-Means Clustering)...")

    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    # Reduce high-dimensional TF-IDF to 50 dimensions using LSA (SVD)
    # This removes noise and helps KMeans find meaningful clusters
    print("Reducing TF-IDF dimensionality via LSA (SVD) for better clustering...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X_train_tfidf)
    X_reduced = normalize(
        X_reduced
    )  # Normalizing often improves KMeans text clustering

    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_reduced)

    score = silhouette_score(
        X_reduced, cluster_labels, sample_size=10000, random_state=42
    )
    print(f"K-Means Silhouette Score (on reduced space): {score:.4f}\n")


def train_models():
    train_df = load_data()

    if train_df is None:
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

    print("Fitting Vectorizer and Transforming Training Data...")
    X_train_tfidf = vectorizer.fit_transform(train_df["combined_text"].fillna(""))
    y_train = train_df["label"]

    print("Computing Word Overlap Features for Training Data...")
    train_overlap = train_df.apply(
        lambda r: get_overlap_ratio(r["article"], r["option"]), axis=1
    ).values.reshape(-1, 1)
    X_train = hstack([X_train_tfidf, csr_matrix(train_overlap)])

    run_unsupervised(X_train_tfidf)

    print("Training Logistic Regression Model...")
    lr_model = LogisticRegression(
        C=5.0, max_iter=1000, random_state=42, class_weight="balanced"
    )
    lr_model.fit(X_train, y_train)

    print("Training Calibrated Linear SVM Model...")
    base_svm = LinearSVC(C=1.0, random_state=42, dual="auto", class_weight="balanced")
    # CalibratedClassifierCV converts SVM distances to probabilities for the ensemble
    svm_model = CalibratedClassifierCV(estimator=base_svm, cv=3)
    svm_model.fit(X_train, y_train)

    print("Training Ensemble (Soft Voting) Model...")
    ensemble_model = VotingClassifier(
        estimators=[("lr", lr_model), ("svm", svm_model)], voting="soft"
    )
    ensemble_model.fit(X_train, y_train)

    os.makedirs("models/model_a", exist_ok=True)
    print("Saving Models and Vectorizer...")
    joblib.dump(vectorizer, "models/model_a/vectorizer.pkl")
    joblib.dump(lr_model, "models/model_a/lr_model.pkl")
    joblib.dump(svm_model, "models/model_a/svm_model.pkl")
    joblib.dump(ensemble_model, "models/model_a/ensemble_model.pkl")
    print("Training complete. Run src/evaluate.py to see metrics.")


if __name__ == "__main__":
    train_models()
