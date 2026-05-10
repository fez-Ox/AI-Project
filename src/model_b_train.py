"""
Model B Training Script
-----------------------
Trains three components:
  1. Word2Vec embeddings on the RACE corpus
  2. A Logistic Regression distractor ranker
  3. A Logistic Regression hint sentence scorer
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import nltk
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def train_word2vec(articles):
    """Train a Word2Vec model on tokenized sentences from the articles."""
    from gensim.models import Word2Vec

    print(f"Found {len(articles)} unique articles.")
    print("Tokenizing corpus (this may take a moment)...")

    sentences = []
    for article in articles:
        for sent in nltk.sent_tokenize(str(article)):
            sentences.append(nltk.word_tokenize(sent.lower()))

    print(f"Training Global Word2Vec model on {len(sentences)} sentences...")
    model = Word2Vec(
        sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=10
    )
    return model


def build_distractor_features(article, candidate, correct_answer, stop_words):
    """Compute simple features for a distractor candidate."""
    art_words = set(str(article).lower().split())
    cand_words = set(str(candidate).lower().split())
    ans_words = set(str(correct_answer).lower().split())

    # Feature 1: word overlap between candidate and article
    overlap_with_article = (
        len(cand_words & art_words) / len(cand_words) if cand_words else 0
    )
    # Feature 2: word overlap between candidate and correct answer
    overlap_with_answer = (
        len(cand_words & ans_words) / len(cand_words) if cand_words else 0
    )
    # Feature 3: length ratio (candidate vs answer)
    length_ratio = len(candidate) / max(len(correct_answer), 1)
    # Feature 4: is candidate a content word (not a stopword)?
    is_content = 1.0 if candidate.lower() not in stop_words else 0.0

    return [overlap_with_article, overlap_with_answer, length_ratio, is_content]


def train_distractor_ranker(df):
    """Train a simple LR to rank distractor candidates (is this a good distractor?)."""
    print("Training distractor ranker...")
    stop_words = set(nltk.corpus.stopwords.words("english"))

    X_features, y_labels = [], []

    # Sample a subset for speed
    sample = df.drop_duplicates(subset="id").head(2000)

    for _, row in sample.iterrows():
        article = str(row["article"])
        question = str(row["question"])
        correct_ans = str(row.get("answer", "")).strip().upper()

        options = {
            "A": str(row.get("A", "")),
            "B": str(row.get("B", "")),
            "C": str(row.get("C", "")),
            "D": str(row.get("D", "")),
        }
        correct_text = options.get(correct_ans, "")

        for key, opt_text in options.items():
            if key == correct_ans:
                continue  # Skip the correct answer
            feats = build_distractor_features(
                article, opt_text, correct_text, stop_words
            )
            X_features.append(feats)
            y_labels.append(1)  # This IS a real distractor

        # Add some negative examples: random words from the article that are NOT distractors
        art_words = [
            w for w in article.split() if w.lower() not in stop_words and len(w) > 2
        ]
        for neg_word in art_words[:3]:
            feats = build_distractor_features(article, neg_word, correct_text, stop_words)
            X_features.append(feats)
            y_labels.append(0)  # Not a real distractor

    X = np.array(X_features)
    y = np.array(y_labels)

    ranker = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    ranker.fit(X, y)
    print(f"Distractor ranker trained on {len(y)} samples. Accuracy: {ranker.score(X, y):.4f}")
    return ranker


def build_hint_features(sentence, question, position, total_sentences):
    """Compute features for hint sentence scoring."""
    sent_words = set(sentence.lower().split())
    q_words = set(question.lower().split())

    # Feature 1: keyword overlap with question
    overlap = len(sent_words & q_words) / max(len(q_words), 1)
    # Feature 2: normalized position in passage (0 = start, 1 = end)
    norm_pos = position / max(total_sentences - 1, 1)
    # Feature 3: sentence length (word count)
    sent_len = len(sentence.split())

    return [overlap, norm_pos, sent_len]


def train_hint_scorer(df):
    """Train a simple LR to score sentences as hints (higher = more relevant)."""
    print("Training hint sentence scorer...")

    X_features, y_scores = [], []

    sample = df.drop_duplicates(subset="id").head(2000)

    for _, row in sample.iterrows():
        article = str(row["article"])
        question = str(row["question"])
        correct_ans = str(row.get("answer", "")).strip().upper()
        correct_text = str(row.get(correct_ans, "")).lower()

        sentences = nltk.sent_tokenize(article)
        if len(sentences) < 2:
            continue

        for i, sent in enumerate(sentences):
            feats = build_hint_features(sent, question, i, len(sentences))
            # Relevance label: 1 if the sentence contains the answer text, else 0
            label = 1 if correct_text and correct_text in sent.lower() else 0
            X_features.append(feats)
            y_scores.append(label)

    X = np.array(X_features)
    y = np.array(y_scores)

    scorer = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    scorer.fit(X, y)
    print(f"Hint scorer trained on {len(y)} samples. Accuracy: {scorer.score(X, y):.4f}")
    return scorer


def main():
    raw_path = "data/raw/train.csv"
    if not os.path.exists(raw_path):
        print(f"Raw training data not found at {raw_path}.")
        return

    print("Loading training data for Model B...")
    df = pd.read_csv(raw_path)

    # 1. Word2Vec
    unique_articles = df["article"].dropna().unique()
    w2v_model = train_word2vec(unique_articles)

    os.makedirs("models/model_b", exist_ok=True)
    w2v_model.save("models/model_b/word2vec.model")
    print("Model B (Word2Vec) saved to models/model_b/word2vec.model!")

    # 2. Distractor ranker
    distractor_ranker = train_distractor_ranker(df)
    joblib.dump(distractor_ranker, "models/model_b/distractor_ranker.pkl")
    print("Distractor ranker saved.")

    # 3. Hint scorer
    hint_scorer = train_hint_scorer(df)
    joblib.dump(hint_scorer, "models/model_b/hint_scorer.pkl")
    print("Hint scorer saved.")

    print("\nModel B training complete!")


if __name__ == "__main__":
    main()
