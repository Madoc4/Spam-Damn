import math

def naive_bayes(email, spam_word_counts, ham_word_counts, n_spam, n_ham, vocab):
    """
    Predict whether an email is spam or not using a Naive Bayes classifier.

    Parameters:
    ----------
    email_words : list of str
        A list of words from the email after preprocessing (e.g., tokenization, stop-word removal).
    
    spam_word_freqs : dict
        A dictionary where keys are words and values are their frequency in spam emails.

    ham_word_freqs : dict
        A dictionary where keys are words and values are their frequency in non-spam (ham) emails.

    spam_prior : float
        Prior probability of an email being spam (P(spam)).

    ham_prior : float
        Prior probability of an email being ham (P(ham)).

    vocabulary_size : int
        Total number of unique words in the vocabulary (used for Laplace smoothing).

    Returns:
    -------
    str
        The predicted label: either "spam" or "ham".

    """
    n_total = n_spam + n_ham

    p_spam = n_spam/n_total
    p_ham = n_ham/n_ntotal

    spam_total_words = sum(spam_word_counts.values())
    ham_total_words = sum(ham_word_counts.values())
    vocab_size = len(vocab)

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
