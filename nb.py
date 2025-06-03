import math


def naive_bayes(
    email: list[str],
    spam_word_counts: dict[str, int],
    ham_word_counts: dict[str, int],
    n_spam: int,
    n_ham: int,
) -> str:
    """
    Predict whether an email is spam or not using a Naive Bayes classifier.

    Returns the predicted label: either "spam" or "ham".
    """
    n_total = n_spam + n_ham

    p_spam = n_spam / n_total
    p_ham = n_ham / n_total

    spam_total_words = sum(spam_word_counts.values())
    ham_total_words = sum(ham_word_counts.values())
    vocab_size = len(set(spam_word_counts) | set(ham_word_counts))

    # use log here
    prob_spam = math.log(p_spam)
    prob_ham = math.log(p_ham)

    for word in email:
        # Laplace Smoothing
        spam_word_freq = spam_word_counts.get(word, 0) + 1
        ham_word_freq = ham_word_counts.get(word, 0) + 1

        prob_spam += math.log(spam_word_freq / (spam_total_words + vocab_size))
        prob_ham += math.log(ham_word_freq / (ham_total_words + vocab_size))

    return "spam" if prob_spam > prob_ham else "ham"
