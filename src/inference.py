import os

import joblib
import nltk
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure punkt and stopwords are downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Global variables to hold loaded models
_vectorizer = None
_model = None
_w2v_model = None


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

    from scipy.sparse import csr_matrix, hstack

    from src.utils import get_overlap_ratio

    # Format inputs exactly as in training
    combined_texts = [f"{article} {article} {question} {opt}" for opt in opt_list]

    # Vectorize
    X_infer_tfidf = _vectorizer.transform(combined_texts)

    # Calculate extra custom features
    overlap_feats = [get_overlap_ratio(article, opt) for opt in opt_list]
    X_infer = hstack(
        [X_infer_tfidf, csr_matrix(np.array(overlap_feats).reshape(-1, 1))]
    )

    # Predict probabilities (if using Logistic Regression) or decision function (if SVM)
    if hasattr(_model, "predict_proba"):
        raw_scores = _model.predict_proba(X_infer)[
            :, 1
        ]  # Probability of class 1 (Correct)
    else:
        raw_scores = _model.decision_function(X_infer)

    # Apply Softmax to convert independent scores into a unified probability distribution that sums to 1
    exp_scores = np.exp(
        raw_scores - np.max(raw_scores)
    )  # Subtracted max for numerical stability
    softmax_probs = exp_scores / exp_scores.sum()

    # Find the index of the option with the highest score
    best_idx = np.argmax(softmax_probs)

    return {
        "predicted_option": opt_keys[best_idx],
        "predicted_text": opt_list[best_idx],
        "confidence_scores": {
            opt_keys[i]: float(softmax_probs[i]) for i in range(len(opt_keys))
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
        temp_vec = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = temp_vec.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            best_idx = np.argmax(sentence_scores)
            best_sentence = sentences[best_idx]
        except ValueError:
            best_sentence = sentences[0]

    words = nltk.word_tokenize(best_sentence)
    if len(words) > 3:
        question = "What is the significance of " + " ".join(words[1:])
        if not question.endswith("?"):
            question = question.rstrip(".!?,") + "?"
    else:
        question = "What does this mean: " + best_sentence

    return question


def generate_distractors(article, question, answer):
    """
    Generates 3 plausible distractors using Gensim's Word2Vec for semantic similarity.
    Applies the "medium similarity heuristic" to avoid synonyms.
    """
    global _w2v_model

    # Load global model if not already loaded
    if _w2v_model is None:
        model_path = "models/model_b/word2vec.model"
        if os.path.exists(model_path):
            _w2v_model = Word2Vec.load(model_path)
        else:
            raise FileNotFoundError(
                "Global Word2Vec model not found. Run model_b_train.py first."
            )

    # Extract potential candidate words from the article
    article_tokens = nltk.word_tokenize(article.lower())
    ans_tokens = nltk.word_tokenize(answer.lower())

    # Tokens from the answer that actually exist in our global vocabulary
    valid_ans_tokens = [t for t in ans_tokens if t in _w2v_model.wv]

    stop_words = set(nltk.corpus.stopwords.words("english"))

    # Extract unique candidates from the article that are valid and not part of the answer
    unique_article_words = list(set(article_tokens))
    valid_candidates = [
        w
        for w in unique_article_words
        if w not in stop_words
        and len(w) > 2
        and w not in ans_tokens
        and w in _w2v_model.wv
    ]

    candidates = []
    if valid_ans_tokens and valid_candidates:
        similarities = []
        for w in valid_candidates:
            # Average similarity of this candidate to the correct answer tokens
            sim = np.mean(
                [_w2v_model.wv.similarity(w, ans_t) for ans_t in valid_ans_tokens]
            )
            similarities.append((w, sim))

        # Rank by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Medium Similarity Heuristic: Drop the top 15% (too similar/synonyms)
        start_idx = max(1, int(len(similarities) * 0.15))
        pool = similarities[start_idx:]

        candidates = [word for word, sim in pool[:3]]

    # Fallback if the pool is too small
    if len(candidates) < 3:
        candidates.extend([w for w in valid_candidates if w not in candidates][:3])
        candidates = list(set(candidates))[:3]

    distractors = [c.capitalize() for c in candidates]

    # Ensure exactly 3 are returned
    while len(distractors) < 3:
        distractors.append("None of the above")

    return distractors[:3]


def generate_hints(article, question):
    """
    Generates graduated hints by ranking article sentences against the question
    using standard Scikit-Learn TF-IDF and Cosine Similarity.
    """
    sentences = nltk.sent_tokenize(article)

    # Degrade gracefully if the article is extremely short
    if len(sentences) < 3:
        return {
            "Hint 1 (Low Detail)": "Focus on the main subject.",
            "Hint 2 (Medium Detail)": sentences[0] if sentences else "",
            "Hint 3 (Near-Explicit)": sentences[-1] if sentences else "",
        }

    # Vectorize sentences and the question
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        q_vec = vectorizer.transform([question])
    except ValueError:
        return {
            "Hint 1 (Low Detail)": "",
            "Hint 2 (Medium Detail)": "",
            "Hint 3 (Near-Explicit)": "",
        }

    # Compute similarity between question and every sentence
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # Rank sentences based on similarity (indices of highest to lowest)
    ranked_indices = np.argsort(sims)[::-1]

    hint3 = sentences[ranked_indices[0]]  # Rank 1: Highest similarity (Near-explicit)
    hint2 = sentences[ranked_indices[1]]  # Rank 2: Medium similarity

    # Low detail: Pick a sentence from the middle of the rankings
    mid_idx = ranked_indices[len(ranked_indices) // 2]
    hint1 = sentences[mid_idx]

    return {
        "Hint 1 (Low Detail)": hint1,
        "Hint 2 (Medium Detail)": hint2,
        "Hint 3 (Near-Explicit)": hint3,
    }
