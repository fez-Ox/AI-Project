import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import nltk
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import get_overlap_ratio, get_word_count, get_option_position

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Cached model references
_vectorizer = None
_model = None
_scaler = None
_distractor_ranker = None
_hint_scorer = None


def load_models(use_ensemble=True):
    """Load Model A vectorizer, scaler, + classifier."""
    global _vectorizer, _model, _scaler

    vec_path = "models/model_a/vectorizer.pkl"
    scaler_path = "models/model_a/scaler.pkl"
    if use_ensemble:
        model_path = "models/model_a/ensemble_model.pkl"
        if not os.path.exists(model_path):
            model_path = "models/model_a/lr_model.pkl"
    else:
        model_path = "models/model_a/lr_model.pkl"

    if os.path.exists(vec_path) and os.path.exists(model_path):
        _vectorizer = joblib.load(vec_path)
        _model = joblib.load(model_path)
        if os.path.exists(scaler_path):
            _scaler = joblib.load(scaler_path)
        return True
    return False


def load_model_b():
    """Load Model B artifacts (distractor ranker and hint scorer)."""
    global _distractor_ranker, _hint_scorer

    ranker_path = "models/model_b/distractor_ranker.pkl"
    scorer_path = "models/model_b/hint_scorer.pkl"

    if os.path.exists(ranker_path):
        _distractor_ranker = joblib.load(ranker_path)
    if os.path.exists(scorer_path):
        _hint_scorer = joblib.load(scorer_path)


# ---------------------------------------------------------------------------
# Model A — Answer Prediction
# ---------------------------------------------------------------------------

def predict_answer(article, question, options, use_ensemble=True):
    """Predict the correct answer option given an article, question, and 4 options."""
    if _vectorizer is None or _model is None:
        if not load_models(use_ensemble=use_ensemble):
            raise FileNotFoundError("Models not found. Please train Model A first.")

    if isinstance(options, dict):
        opt_list = list(options.values())
        opt_keys = list(options.keys())
    else:
        opt_list = options
        opt_keys = ["A", "B", "C", "D"]

    combined_texts = [f"{article} {question} {opt}" for opt in opt_list]

    X_tfidf = _vectorizer.transform(combined_texts)
    
    # Handcrafted features: article_len, question_len, option_len, overlap_ratio, option_position
    art_len = get_word_count(article)
    q_len = get_word_count(question)
    
    handcrafted = []
    for opt in opt_list:
        handcrafted.append([
            art_len,
            q_len,
            get_word_count(opt),
            get_overlap_ratio(article, opt),
            get_option_position(article, opt)
        ])
        
    X_handcrafted = np.array(handcrafted)
    if _scaler is not None:
        X_handcrafted = _scaler.transform(X_handcrafted)
        
    X_infer = hstack([X_tfidf, csr_matrix(X_handcrafted)])

    if hasattr(_model, "predict_proba"):
        raw_scores = _model.predict_proba(X_infer)[:, 1]
    else:
        raw_scores = _model.decision_function(X_infer)

    # Softmax for a unified probability distribution
    exp_scores = np.exp(raw_scores - np.max(raw_scores))
    softmax_probs = exp_scores / exp_scores.sum()

    best_idx = np.argmax(softmax_probs)

    return {
        "predicted_option": opt_keys[best_idx],
        "predicted_text": opt_list[best_idx],
        "confidence_scores": {
            opt_keys[i]: float(softmax_probs[i]) for i in range(len(opt_keys))
        },
    }


# ---------------------------------------------------------------------------
# Model A — Question Generation (Wh-word templates + ML ranking)
# ---------------------------------------------------------------------------

WH_TEMPLATES = [
    ("who", "Who {verb_phrase}?"),
    ("what", "What {verb_phrase}?"),
    ("where", "Where {verb_phrase}?"),
    ("when", "When {verb_phrase}?"),
    ("why", "Why {verb_phrase}?"),
]


def _pick_wh_template(sentence):
    """Pick the best Wh-word template based on sentence content."""
    lower = sentence.lower()
    if any(w in lower for w in ["because", "reason", "due to"]):
        return "Why"
    if any(w in lower for w in ["where", "location", "place", "city", "country"]):
        return "Where"
    if any(w in lower for w in ["when", "year", "date", "time", "day", "month"]):
        return "When"
    if any(w in lower for w in ["he ", "she ", "they ", "mr.", "mrs.", "dr."]):
        return "Who"
    return "What"


def generate_question(article):
    """Generate a question from the article using Wh-word templates."""
    sentences = nltk.sent_tokenize(article)
    if not sentences:
        return "Could not extract sentences from the article."

    # Step 1: Score sentences by keyword importance
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

    # Step 2: Apply Wh-word template
    wh_word = _pick_wh_template(best_sentence)
    words = nltk.word_tokenize(best_sentence)

    if len(words) > 3:
        # Remove the first noun/subject and turn into a question
        verb_phrase = " ".join(words[1:]).rstrip(".!?,;:")
        question = f"{wh_word} {verb_phrase}?"
    else:
        question = f"{wh_word} does this mean: {best_sentence}"

    return question


# ---------------------------------------------------------------------------
# Model B — Distractor Generation
# ---------------------------------------------------------------------------

def generate_distractors(article, question, answer):
    """
    Generate 3 plausible distractors using TF-IDF cosine similarity
    with a diversity penalty and optional ML ranker.
    """
    # Load ML ranker if available
    if _distractor_ranker is None:
        load_model_b()

    article_tokens = nltk.word_tokenize(article.lower())
    ans_tokens = set(nltk.word_tokenize(answer.lower()))
    stop_words = set(nltk.corpus.stopwords.words("english"))

    # Step 1: Extract candidate words from the passage
    valid_candidates = list(set(
        w for w in article_tokens
        if w not in stop_words and len(w) > 2 and w not in ans_tokens
    ))

    candidates = []
    if valid_candidates:
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            vectorizer.fit([article])
            ans_vec = vectorizer.transform([answer])
            cand_vecs = vectorizer.transform(valid_candidates)
            sims = cosine_similarity(ans_vec, cand_vecs).flatten()

            scored = [(valid_candidates[i], sims[i]) for i in range(len(valid_candidates))]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Medium similarity heuristic: skip top 15% (too similar)
            start_idx = max(1, int(len(scored) * 0.15))
            pool = scored[start_idx:]

            # Step 2: Apply diversity penalty — pick candidates that differ from each other
            selected = []
            for word, sim in pool:
                if len(selected) >= 3:
                    break
                # Check diversity: don't pick words too similar to already selected
                if all(word[:3] != s[:3] for s in selected):
                    selected.append(word)

            candidates = selected

        except ValueError:
            pass

    # Fallback if pool is too small
    if len(candidates) < 3:
        remaining = [w for w in valid_candidates if w not in candidates]
        random.shuffle(remaining)
        candidates.extend(remaining[: 3 - len(candidates)])

    # If ML ranker is available, re-rank candidates
    if _distractor_ranker is not None and len(candidates) > 3:
        features = []
        for c in candidates:
            art_words = set(article.lower().split())
            cand_words = set(c.lower().split())
            ans_words_set = set(answer.lower().split())
            ov_art = len(cand_words & art_words) / max(len(cand_words), 1)
            ov_ans = len(cand_words & ans_words_set) / max(len(cand_words), 1)
            len_ratio = len(c) / max(len(answer), 1)
            is_content = 1.0 if c.lower() not in stop_words else 0.0
            features.append([ov_art, ov_ans, len_ratio, is_content])

        probs = _distractor_ranker.predict_proba(np.array(features))[:, 1]
        ranked = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
        candidates = [w for w, _ in ranked[:3]]

    distractors = [c.capitalize() for c in candidates[:3]]

    while len(distractors) < 3:
        distractors.append("None of the above")

    return distractors[:3]


# ---------------------------------------------------------------------------
# Model B — Hint Generation (extractive with ML scoring)
# ---------------------------------------------------------------------------

def generate_hints(article, question):
    """Generate graduated hints from the passage, optionally scored by ML."""
    if _hint_scorer is None:
        load_model_b()

    sentences = nltk.sent_tokenize(article)

    if len(sentences) < 3:
        return {
            "Hint 1 (Low Detail)": "Focus on the main subject.",
            "Hint 2 (Medium Detail)": sentences[0] if sentences else "",
            "Hint 3 (Near-Explicit)": sentences[-1] if sentences else "",
        }

    # Score sentences using ML model if available, else fallback to TF-IDF cosine
    if _hint_scorer is not None:
        features = []
        for i, sent in enumerate(sentences):
            q_words = set(question.lower().split())
            s_words = set(sent.lower().split())
            overlap = len(s_words & q_words) / max(len(q_words), 1)
            norm_pos = i / max(len(sentences) - 1, 1)
            sent_len = len(sent.split())
            features.append([overlap, norm_pos, sent_len])

        scores = _hint_scorer.predict_proba(np.array(features))[:, 1]
    else:
        # Fallback: TF-IDF cosine similarity
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            q_vec = vectorizer.transform([question])
            scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        except ValueError:
            return {
                "Hint 1 (Low Detail)": "",
                "Hint 2 (Medium Detail)": "",
                "Hint 3 (Near-Explicit)": "",
            }

    ranked_indices = np.argsort(scores)[::-1]

    hint3 = sentences[ranked_indices[0]]  # Most relevant
    hint2 = sentences[ranked_indices[1]]  # Medium relevance
    mid_idx = ranked_indices[len(ranked_indices) // 2]
    hint1 = sentences[mid_idx]  # Low detail

    return {
        "Hint 1 (Low Detail)": hint1,
        "Hint 2 (Medium Detail)": hint2,
        "Hint 3 (Near-Explicit)": hint3,
    }
