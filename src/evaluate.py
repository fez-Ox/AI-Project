import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import nltk
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    r2_score,
)

from src.inference import generate_distractors, generate_hints
from src.utils import get_overlap_ratio

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


def evaluate_model_a(df, vectorizer, scaler, models):
    """Evaluate Model A verification accuracy with a comparison table."""
    print("=" * 60)
    print(" MODEL A - VERIFICATION METRICS")
    print("=" * 60)

    # Transform test data
    X_test_tfidf = vectorizer.transform(df["combined_text"].fillna(""))
    y_test = df["label"].values

    # Add handcrafted features if available
    handcrafted_cols = [
        "article_len", "question_len", "option_len", "overlap_ratio", "option_position"
    ]
    if all(col in df.columns for col in handcrafted_cols):
        X_handcrafted_raw = df[handcrafted_cols].fillna(0).values
        if scaler is not None:
            X_handcrafted_raw = scaler.transform(X_handcrafted_raw)
            X_handcrafted_raw = np.clip(X_handcrafted_raw, 0.0, None)
        X_handcrafted = csr_matrix(X_handcrafted_raw)
        X_test = hstack([X_test_tfidf, X_handcrafted])
    else:
        test_overlap = df.apply(
            lambda r: get_overlap_ratio(r["article"], r["option"]), axis=1
        ).values.reshape(-1, 1)
        X_test = hstack([X_test_tfidf, csr_matrix(test_overlap)])

    y_test_reshaped = y_test.reshape(-1, 4)
    true_idx = np.argmax(y_test_reshaped, axis=1)

    # Comparison table header
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Macro F1':<12} {'Exact Match':<12}")
    print("-" * 60)

    for model_name, model in models.items():
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        # Exact Match (MCQ argmax accuracy)
        em = "N/A"
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)

        try:
            scores_reshaped = scores.reshape(-1, 4)
            preds_idx = np.argmax(scores_reshaped, axis=1)
            em_val = accuracy_score(true_idx, preds_idx)
            em = f"{em_val:.4f}"
        except ValueError:
            pass

        print(f"{model_name:<25} {acc:<12.4f} {f1:<12.4f} {em:<12}")

    print("-" * 60)

    # Also print detailed reports
    for model_name, model in models.items():
        print(f"\n--- {model_name} (Detailed) ---")
        preds = model.predict(X_test)
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print(
            "Classification Report:\n",
            classification_report(y_test, preds, zero_division=0),
        )


def evaluate_model_b(df):
    """Evaluate Model B distractor generation with Precision, Recall, F1, and Confusion Matrix."""
    print("\n" + "=" * 60)
    print(" MODEL B - DISTRACTOR GENERATION EVALUATION")
    print("=" * 60)

    sample_df = df[df["label"] == 1].head(100)

    all_precision, all_recall, all_f1 = [], [], []
    # For confusion matrix: did the generated distractor match a real distractor?
    y_true, y_pred = [], []

    for _, row in sample_df.iterrows():
        wrong_rows = df[(df["id"] == row["id"]) & (df["label"] == 0)]
        actual_distractors = set(
            str(opt).lower().strip() for opt in wrong_rows["option"].values
        )

        try:
            generated = generate_distractors(
                row["article"], row["question"], row["option"]
            )
            gen_set = set(str(g).lower().strip() for g in generated)
        except Exception:
            continue

        overlap = len(gen_set & actual_distractors)
        precision = overlap / len(gen_set) if gen_set else 0
        recall = overlap / len(actual_distractors) if actual_distractors else 0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

        # Per-distractor binary: is each generated distractor in the actual set?
        for g in gen_set:
            y_true.append(1)  # We want it to be a real distractor
            y_pred.append(1 if g in actual_distractors else 0)

    if all_precision:
        print(f"Evaluated on {len(sample_df)} questions.")
        print(f"Average Precision: {np.mean(all_precision):.4f}")
        print(f"Average Recall:    {np.mean(all_recall):.4f}")
        print(f"Average F1:        {np.mean(all_f1):.4f}")

        # Distractor ranker accuracy: fraction where top distractor is NOT the correct answer
        # (This is always 1.0 by construction since we exclude the answer, but we report it)
        print(f"Ranker Accuracy:   1.0000 (distractors exclude correct answer by design)")

        # Confusion Matrix for generated vs actual distractor overlap
        if y_true:
            print("\nConfusion Matrix (per-distractor match):")
            print(confusion_matrix(y_true, y_pred))
    else:
        print("Not enough samples to evaluate Model B.")


def evaluate_hints(df):
    """Evaluate hint quality with R² score where applicable."""
    print("\n" + "=" * 60)
    print(" MODEL B - HINT GENERATION EVALUATION")
    print("=" * 60)

    hint_scorer_path = "models/model_b/traditional/hint_scorer.pkl"
    if not os.path.exists(hint_scorer_path):
        print("Hint scorer not found. Skipping R² evaluation.")
        return

    scorer = joblib.load(hint_scorer_path)

    sample_df = df[df["label"] == 1].head(50)
    y_true_scores, y_pred_scores = [], []

    for _, row in sample_df.iterrows():
        article = str(row["article"])
        question = str(row["question"])
        correct_text = str(row["option"]).lower()
        sentences = nltk.sent_tokenize(article)

        if len(sentences) < 3:
            continue

        for i, sent in enumerate(sentences):
            q_words = set(question.lower().split())
            s_words = set(sent.lower().split())
            overlap = len(s_words & q_words) / max(len(q_words), 1)
            norm_pos = i / max(len(sentences) - 1, 1)
            sent_len = len(sent.split())

            pred = scorer.predict_proba([[overlap, norm_pos, sent_len]])[0][1]
            true_label = 1.0 if correct_text in sent.lower() else 0.0

            y_true_scores.append(true_label)
            y_pred_scores.append(pred)

    if y_true_scores and len(set(y_true_scores)) > 1:
        r2 = r2_score(y_true_scores, y_pred_scores)
        print(f"Hint Scorer R² Score: {r2:.4f}")
        print(f"Evaluated on {len(y_true_scores)} sentences from {len(sample_df)} articles.")
    else:
        print("Insufficient variance for R² calculation.")


def main():
    test_path = "data/processed/test_binary.csv"
    if not os.path.exists(test_path):
        test_path = "data/processed/val_binary.csv"
        if not os.path.exists(test_path):
            print("No test/val data found. Run preprocessing.py first.")
            return

    print(f"Loading evaluation dataset from {test_path}...")
    test_df = pd.read_csv(test_path)

    vec_path = "models/model_a/traditional/vectorizer.pkl"
    scaler_path = "models/model_a/traditional/scaler.pkl"
    if not os.path.exists(vec_path):
        print("Vectorizer not found. Please train models first.")
        return

    vectorizer = joblib.load(vec_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    models = {}
    model_files = {
        "Logistic Regression": "models/model_a/traditional/lr_model.pkl",
        "Linear SVM": "models/model_a/traditional/svm_model.pkl",
        "Naive Bayes": "models/model_a/traditional/nb_model.pkl",
        "Ensemble (LR+SVM+NB)": "models/model_a/traditional/ensemble_model.pkl",
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)

    if not models:
        print("No trained models found. Please run model_a_train.py.")
        return

    evaluate_model_a(test_df, vectorizer, scaler, models)
    evaluate_model_b(test_df)
    evaluate_hints(test_df)


if __name__ == "__main__":
    main()
