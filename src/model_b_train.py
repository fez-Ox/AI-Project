import os

import joblib
import nltk
import pandas as pd
from gensim.models import Word2Vec

# Ensure punkt is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def train_model_b():
    print("Loading training data for Model B...")
    # Load raw train data to get unique articles
    train_path = "data/raw/train.csv"
    if not os.path.exists(train_path):
        print(
            f"Dataset not found at {train_path}. Please place it there to train Model B."
        )
        return

    df = pd.read_csv(train_path)
    # Get unique articles to avoid redundant processing
    articles = df["article"].dropna().unique()
    print(f"Found {len(articles)} unique articles.")

    print("Tokenizing corpus (this may take a moment)...")
    corpus_sentences = []
    for article in articles:
        # Split article into sentences, then words, lowercased
        for sentence in nltk.sent_tokenize(article):
            corpus_sentences.append(nltk.word_tokenize(sentence.lower()))

    print(f"Training Global Word2Vec model on {len(corpus_sentences)} sentences...")
    # Train Word2Vec: vector_size=100 is standard, min_count=2 removes extreme typos
    w2v_model = Word2Vec(
        sentences=corpus_sentences, vector_size=100, window=5, min_count=2, workers=4
    )

    os.makedirs("models/model_b", exist_ok=True)
    model_path = "models/model_b/word2vec.model"
    w2v_model.save(model_path)
    print(f"Model B (Word2Vec) saved to {model_path}!")


if __name__ == "__main__":
    train_model_b()
