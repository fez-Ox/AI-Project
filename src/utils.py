import re


def clean_text(text):
    """Lowercase, strip whitespace, and remove punctuation."""
    if not text or str(text) == "nan":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text


def get_overlap_ratio(text_a, text_b):
    """Fraction of words in text_b that also appear in text_a."""
    words_a = set(str(text_a).lower().split())
    words_b = set(str(text_b).lower().split())
    if not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_b)


def get_word_count(text):
    """Number of words in text."""
    return len(str(text).split())


def get_option_position(article, option):
    """Normalized position (0-1) of where the option first appears in the article.
    Returns -1 if not found."""
    article_lower = str(article).lower()
    option_lower = str(option).lower().strip()
    pos = article_lower.find(option_lower)
    if pos == -1 or len(article_lower) == 0:
        return -1.0
    return pos / len(article_lower)
