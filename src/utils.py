def get_overlap_ratio(article, option):
    """
    Calculates the percentage of words in the option that also appear in the article.
    This explicit numerical feature helps the classical ML model easily spot
    options that have high lexical overlap with the passage.
    """
    art_words = set(str(article).lower().split())
    opt_words = set(str(option).lower().split())

    if not opt_words:
        return 0.0

    # Return the ratio of overlapping words
    return len(art_words.intersection(opt_words)) / len(opt_words)
