import math


def naive_bayes(
    email: list[str],
    positive_word_counts: dict[str, int],
    negative_word_counts: dict[str, int],
    n_positive: int,
    n_negative: int,
) -> bool:
    """
    Predict whether an email is positive or negative using a Naive Bayes classifier.

    Returns a boolean indicating if it is positve (True).
    """
    n_total = n_positive + n_negative

    p_positive = n_positive / n_total
    p_negative = n_negative / n_total

    positive_total_words = sum(positive_word_counts.values())
    negative_total_words = sum(negative_word_counts.values())
    vocab_size = len(set(positive_word_counts) | set(negative_word_counts))

    # use log here
    prob_positive = math.log(p_positive)
    prob_negative = math.log(p_negative)

    for word in email:
        # Laplace Smoothing
        positive_word_freq = positive_word_counts.get(word, 0) + 1
        negative_word_freq = negative_word_counts.get(word, 0) + 1

        prob_positive += math.log(
            positive_word_freq / (positive_total_words + vocab_size)
        )
        prob_negative += math.log(
            negative_word_freq / (negative_total_words + vocab_size)
        )

    return prob_positive > prob_negative
