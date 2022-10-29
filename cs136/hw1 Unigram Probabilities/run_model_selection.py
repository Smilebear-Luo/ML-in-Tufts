'''
Summary
-------
This script produces a figure showing how the training set evidence varies (y-axis) as we
consider different alpha values (x-axis) for the Dirichlet prior of our model.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Vocabulary import Vocabulary
from PosteriorPredictiveEstimator import PosteriorPredictiveEstimator

from scipy.special import gammaln
from scipy.special import gamma


def evaluate_log_evidence(estimator, word_list):
    ''' Evaluate the log of the evidence

    Assumes the Dirichlet-Multinomial model, marginalizing out the parameter vector

    Args
    ----
    estimator : PosteriorPredictiveEstimator
            Defines a Dir-Mult model
    word_list : list of strings
            Assumed that each string is in the vocabulary of the estimator

    Returns
    -------
    log_proba : scalar float
            Represents value of p(word_list | alpha)
            This marginalizes out the probability parameters

    Examples
    --------
    >>> est = PosteriorPredictiveEstimator(vocab=Vocabulary(["a", "b"]), alpha=1.0)
    >>> np.exp(evaluate_log_evidence(est, ["a"]))
    0.5
    '''
    assert isinstance(estimator, PosteriorPredictiveEstimator)
    # TODO Fit the estimator to the words

    estimator.fit(word_list)
    # TODO Calculate the log evidence using provided formulas
    V = estimator.vocab.size
    alpha = estimator.alpha
    N = estimator.total_count
    constant = gammaln(V * alpha) - gammaln(N + V * alpha)
    pai_0 = float(0)
    pai_1 = gammaln(alpha) * V
    for i in range(estimator.vocab.size):
        pai = gammaln(estimator.count_V[i]+estimator.alpha)
        pai_0 = pai+pai_0


    log_evidence = constant + pai_0 - pai_1

    return log_evidence


if __name__ == '__main__':
    vocab = Vocabulary(["../data/training_data.txt", "../data/test_data.txt"])

    # Read in word list from plain-text file
    # The call to strip makes sure we have no words with lead/trailing whitespace
    train_word_list = [str.strip(s)
                       for s in np.loadtxt("../data/training_data.txt", dtype=str, delimiter=' ')]
    test_word_list = [str.strip(s)
                      for s in np.loadtxt("../data/test_data.txt", dtype=str, delimiter=' ')]

    frac_train_list = [1./128, 1./16, 1.]
    n_train_list = [int(np.ceil(frac * len(train_word_list)))
                    for frac in frac_train_list]

    alpha_list = np.logspace(-2, 3, 11)

    fig_handle, ax_grid = plt.subplots(
        nrows=1, ncols=len(n_train_list), figsize=(12, 3),
        squeeze=True, sharex=True, sharey=True)

    for nn, N in enumerate(n_train_list):
        print("Plotting %d/%d with N = %d ..." % (nn, len(n_train_list), N))

        log_evidence_list = np.zeros_like(alpha_list)
        heldout_logproba_list = np.zeros_like(alpha_list)

        # TODO fit an estimator to each alpha value
        # TODO evaluate training set's log evidence at each alpha value
        # TODO evaluate test set's estimated probability with 'score'
        length = n_train_list[nn]
        train_set_temporary = train_word_list[:length]
        for i in range(alpha_list.size):
            ppe = PosteriorPredictiveEstimator(Vocabulary(train_word_list), alpha=alpha_list[i])
            ppe.fit(train_set_temporary)
            log_evidence_list[i] = evaluate_log_evidence(ppe, train_word_list)
            heldout_logproba_list[i] = ppe.score(test_word_list)


        arange_list = np.arange(len(alpha_list))
        ax_grid[nn].plot(arange_list, heldout_logproba_list, 'r.-')
        ax_grid[nn].plot(arange_list, log_evidence_list, 'ks-')

        ax_grid[nn].set_xticks(arange_list[::2])
        ax_grid[nn].set_xticklabels(['% .2g' % a for a in alpha_list[::2]])
        ax_grid[nn].set_title('N = %d' % N)
        ax_grid[nn].set_ylim([-10.0, -8.5])

    plt.show()
