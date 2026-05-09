import os

import joblib
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure punkt is downloaded for sentence tokenization
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Global variables to hold loaded models
_vectorizer = None
_model = None


def load_models():
    """Load the vectorizer and classification model from disk."""
    global _vectorizer, _model

    vec_path = "models/model_a/vectorizer.pkl"
    model_path = "models/model_a/lr_model.pkl"  # Can be switched to svm_model.pkl

    if os.path.exists(vec_path) and os.path.exists(model_path):
        _vectorizer = joblib.load(vec_path)
        _model = joblib.load(model_path)
        return True
    return False


def predict_answer(article, question, options):
    """
    Predicts the correct answer from the given options.
    options: List or Dictionary of 4 text strings.
    """
    if _vectorizer is None or _model is None:
        if not load_models():
            raise FileNotFoundError("Models not found. Please train Model A first.")

    if isinstance(options, dict):
        opt_list = list(options.values())
        opt_keys = list(options.keys())
    else:
        opt_list = options
        opt_keys = ["A", "B", "C", "D"]

    # Format inputs exactly as in training
    combined_texts = [f"{article} {article} {question} {opt}" for opt in opt_list]

    # Vectorize
    X_infer = _vectorizer.transform(combined_texts)

    # Predict probabilities (if using Logistic Regression) or decision function (if SVM)
    if hasattr(_model, "predict_proba"):
        probs = _model.predict_proba(X_infer)[:, 1]  # Probability of class 1 (Correct)
    else:
        probs = _model.decision_function(X_infer)

    # Find the index of the option with the highest score
    best_idx = np.argmax(probs)

    return {
        "predicted_option": opt_keys[best_idx],
        "predicted_text": opt_list[best_idx],
        "confidence_scores": {
            opt_keys[i]: float(probs[i]) for i in range(len(opt_keys))
        },
    }


def generate_question(article):
    """
    Generates a simple question from the article based on TF-IDF and sentence ranking.
    Uses an internal, lightweight vectorizer just for the sentences of this article.
    """
    sentences = nltk.sent_tokenize(article)
    if not sentences:
        return "Could not extract sentences from the article."

    if len(sentences) == 1:
        best_sentence = sentences[0]
    else:
        # Create a mini TF-IDF matrix for just these sentences
        temp_vec = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = temp_vec.fit_transform(sentences)

            # Simple heuristic: The sentence with the highest sum of TF-IDF scores
            # is deemed the most "information-dense" or relevant.
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            best_idx = np.argmax(sentence_scores)
            best_sentence = sentences[best_idx]
        except ValueError:
            # Fallback if vocabulary is too small or empty
            best_sentence = sentences[0]

    # Apply simple template conversion
    # E.g., replace the first named entity or subject with a Wh- word.
    # For this simplified implementation, we'll do a rudimentary transformation.

    words = nltk.word_tokenize(best_sentence)
    if len(words) > 3:
        # Very naive heuristic: Replace the beginning of the sentence
        question = "What is the significance of " + " ".join(words[1:])
        if not question.endswith("?"):
            question = question.rstrip(".!?,") + "?"
    else:
        question = "What does this mean: " + best_sentence

    return question
