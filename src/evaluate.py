import os

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
)

from src.inference import generate_distractors
from src.utils import get_overlap_ratio


def evaluate_model_a(df, vectorizer, models):
    """Evaluates Model A's multiple choice verification accuracy using the provided models."""
    print("=" * 60)
    print(" MODEL A - VERIFICATION METRICS")
    print("=" * 60)

    # Transform test data
    X_test_tfidf = vectorizer.transform(df["combined_text"].fillna(""))
    y_test = df["label"].values

    test_overlap = df.apply(
        lambda r: get_overlap_ratio(r["article"], r["option"]), axis=1
    ).values.reshape(-1, 1)
    X_test = hstack([X_test_tfidf, csr_matrix(test_overlap)])

    # Pre-reshape labels for Argmax evaluation
    y_test_reshaped = y_test.reshape(-1, 4)
    true_idx = np.argmax(y_test_reshaped, axis=1)

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")

        # 1. Standard Binary Classification Metrics
        preds = model.predict(X_test)
        print("Binary Accuracy: {:.4f}".format(accuracy_score(y_test, preds)))
        print("Macro F1: {:.4f}".format(f1_score(y_test, preds, average="macro")))
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print(
            "Classification Report:\n",
            classification_report(y_test, preds, zero_division=0),
        )

        # 2. Exact Match (MCQ Argmax Accuracy)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)

        try:
            scores_reshaped = scores.reshape(-1, 4)
            preds_idx = np.argmax(scores_reshaped, axis=1)
            argmax_acc = accuracy_score(true_idx, preds_idx)
            print(f">>> Exact Match (Real MCQ Accuracy): {argmax_acc:.4f} <<<")
        except ValueError:
            print(
                "Could not perform Exact Match evaluation (dataset length not a multiple of 4)."
            )


def evaluate_model_b(df):
    """
    Provides a heuristic evaluation of Model B's Distractor Generation.
    Calculates precision/recall of the generated distractors overlapping with the dataset's actual distractors.
    Note: Standard R2 score is ignored because it applies strictly to numerical regression.
    """
    print("\n" + "=" * 60)
    print(" MODEL B - DISTRACTOR GENERATION (Heuristic Evaluation)")
    print("=" * 60)

    # We evaluate on a small sample of the test set to avoid excessive runtime
    sample_df = df[df["label"] == 1].head(
        100
    )  # Only rows containing the correct answer

    all_precision = []
    all_recall = []
    all_f1 = []
    exact_matches = 0

    for _, row in sample_df.iterrows():
        # Get actual wrong options from the original multiple-choice problem
        # We find the other 3 rows in the dataframe that share the same 'id' and have label==0
        wrong_rows = df[(df["id"] == row["id"]) & (df["label"] == 0)]
        actual_distractors = set(
            [str(opt).lower().strip() for opt in wrong_rows["option"].values]
        )

        # Generate distractors using Model B
        try:
            generated = generate_distractors(
                row["article"], row["question"], row["option"]
            )
            gen_distractors = set([str(g).lower().strip() for g in generated])
        except FileNotFoundError:
            print(
                "Cannot evaluate Model B without global Word2Vec model. Run src/model_b_train.py"
            )
            return

        # Calculate overlap
        overlap = len(gen_distractors.intersection(actual_distractors))

        if len(gen_distractors) > 0:
            precision = overlap / len(gen_distractors)
            recall = (
                overlap / len(actual_distractors) if len(actual_distractors) > 0 else 0
            )
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

            if overlap == 3:
                exact_matches += 1

    if all_precision:
        print(f"Evaluated on {len(sample_df)} questions.")
        print(f"Average Distractor Precision: {np.mean(all_precision):.4f}")
        print(f"Average Distractor Recall: {np.mean(all_recall):.4f}")
        print(f"Average Distractor F1: {np.mean(all_f1):.4f}")
        print(f"Exact Distractor Set Matches: {exact_matches}")
        print("\nNote: Model B is fully Unsupervised/Generative.")
        print(
            "True 'R2 Score' is not applicable (R2 is exclusively for continuous regression problems)."
        )
    else:
        print("Not enough samples to evaluate Model B.")


def main():
    test_path = "data/processed/test_binary.csv"
    if not os.path.exists(test_path):
        print("Test dataset not found. Using Validation dataset instead.")
        test_path = "data/processed/val_binary.csv"
        if not os.path.exists(test_path):
            print("Validation dataset not found either. Please run preprocessing.py")
            return

    print(f"Loading evaluation dataset from {test_path}...")
    test_df = pd.read_csv(test_path)

    vec_path = "models/model_a/vectorizer.pkl"
    if not os.path.exists(vec_path):
        print("Vectorizer not found. Please train models first.")
        return

    vectorizer = joblib.load(vec_path)

    models = {}
    model_files = {
        "Logistic Regression": "models/model_a/lr_model.pkl",
        "Linear SVM": "models/model_a/svm_model.pkl",
        "Ensemble (Voting)": "models/model_a/ensemble_model.pkl",
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)

    if not models:
        print("No trained models found. Please run model_a_train.py")
        return

    evaluate_model_a(test_df, vectorizer, models)
    evaluate_model_b(test_df)


if __name__ == "__main__":
    main()
