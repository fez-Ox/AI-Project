import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


def load_data():
    path = "data/processed/train_binary.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def compute_purity(cluster_labels, true_labels):
    """Cluster purity: fraction of samples assigned to the majority class per cluster."""
    total = 0
    for cluster_id in set(cluster_labels):
        mask = cluster_labels == cluster_id
        class_counts = np.bincount(true_labels[mask])
        total += class_counts.max()
    return total / len(true_labels)


def run_unsupervised(X_features, y_train):
    """Run K-Means clustering and report silhouette score + purity."""
    print("\n--- Unsupervised: K-Means Clustering ---")

    # Reduce dimensionality for better clustering
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = normalize(svd.fit_transform(X_features))

    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_reduced)

    sil_score = silhouette_score(
        X_reduced, cluster_labels, sample_size=10000, random_state=42
    )
    purity = compute_purity(cluster_labels, y_train.values)

    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Cluster Purity:   {purity:.4f}")
    return sil_score, purity


def train_models():
    train_df = load_data()
    if train_df is None:
        print("Processed data not found. Run preprocessing.py first.")
        return

    y_train = train_df["label"]

    # --- One-Hot Encoding (primary, as per spec) ---
    print("Fitting One-Hot Encoder (CountVectorizer binary=True)...")
    ohe_vectorizer = CountVectorizer(
        binary=True,
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X_train_ohe = ohe_vectorizer.fit_transform(
        train_df["combined_text"].fillna("")
    )

    # --- TF-IDF (optional alternative) ---
    print("Fitting TF-IDF Vectorizer (optional)...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(
        train_df["combined_text"].fillna("")
    )

    # --- Handcrafted features ---
    print("Adding handcrafted lexical features...")
    handcrafted_cols = [
        "article_len", "question_len", "option_len", "overlap_ratio", "option_position"
    ]
    # Fill missing handcrafted features if they don't exist in older data
    for col in handcrafted_cols:
        if col not in train_df.columns:
            train_df[col] = 0.0
    X_handcrafted = csr_matrix(train_df[handcrafted_cols].fillna(0).values)

    # Combined feature sets
    X_ohe_full = hstack([X_train_ohe, X_handcrafted])
    X_tfidf_full = hstack([X_train_tfidf, X_handcrafted])

    # --- Unsupervised clustering ---
    sil, purity = run_unsupervised(X_train_ohe, y_train)

    # --- Supervised models (trained on OHE as primary) ---
    print("\nTraining Logistic Regression (One-Hot)...")
    lr_model = LogisticRegression(
        C=1.0, max_iter=5000, random_state=42, class_weight="balanced", solver="saga"
    )
    lr_model.fit(X_ohe_full, y_train)

    print("Training Calibrated Linear SVM (One-Hot)...")
    base_svm = LinearSVC(C=1.0, random_state=42, dual="auto", class_weight="balanced", max_iter=5000)
    svm_model = CalibratedClassifierCV(estimator=base_svm, cv=3)
    svm_model.fit(X_ohe_full, y_train)

    print("Training Naive Bayes (One-Hot)...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_ohe, y_train)  # NB needs non-negative, so use OHE only

    print("Training Ensemble (Soft Voting: LR + SVM + NB)...")
    ensemble_model = VotingClassifier(
        estimators=[("lr", lr_model), ("svm", svm_model), ("nb", nb_model)],
        voting="soft",
    )
    ensemble_model.fit(X_ohe_full, y_train)

    # --- Comparison table ---
    print("\n" + "=" * 55)
    print(" COMPARISON: Unsupervised vs Supervised")
    print("=" * 55)
    print(f"{'Method':<30} {'Metric':<20} {'Value':<10}")
    print("-" * 55)
    print(f"{'K-Means Clustering':<30} {'Silhouette Score':<20} {sil:.4f}")
    print(f"{'K-Means Clustering':<30} {'Cluster Purity':<20} {purity:.4f}")
    print(f"{'Logistic Regression':<30} {'Train Accuracy':<20} {lr_model.score(X_ohe_full, y_train):.4f}")
    print(f"{'Linear SVM':<30} {'Train Accuracy':<20} {svm_model.score(X_ohe_full, y_train):.4f}")
    print(f"{'Naive Bayes':<30} {'Train Accuracy':<20} {nb_model.score(X_train_ohe, y_train):.4f}")
    print(f"{'Ensemble (LR+SVM+NB)':<30} {'Train Accuracy':<20} {ensemble_model.score(X_ohe_full, y_train):.4f}")
    print("=" * 55)

    # --- Save models ---
    os.makedirs("models/model_a/traditional", exist_ok=True)
    print("\nSaving models and vectorizers...")
    joblib.dump(ohe_vectorizer, "models/model_a/vectorizer.pkl")  # Primary
    joblib.dump(tfidf_vectorizer, "models/model_a/tfidf_vectorizer.pkl")  # Optional
    joblib.dump(lr_model, "models/model_a/lr_model.pkl")
    joblib.dump(svm_model, "models/model_a/svm_model.pkl")
    joblib.dump(nb_model, "models/model_a/nb_model.pkl")
    joblib.dump(ensemble_model, "models/model_a/ensemble_model.pkl")
    print("Training complete. Run src/evaluate.py to see metrics.")


if __name__ == "__main__":
    train_models()
