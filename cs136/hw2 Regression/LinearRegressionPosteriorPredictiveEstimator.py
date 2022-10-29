'''
Summary
-------
Defines the posterior predictive estimator for linear regression

PosteriorPredictiveEstimator supports these API functions common to any sklearn-like regression model:
* fit
* predict
* score

Resources
---------
See COMP 136 course website for the complete problem description and all math details
'''

import numpy as np
import scipy.stats
import math

class LinearRegressionPosteriorPredictiveEstimator():
    """
    Posterior Predictive Estimator for linear regression

    Attributes
    ----------
    feature_transformer : feature transformer
        Any transformer that implements "transform" and "get_feature_size"
        See provided FeatureTransformPolynomial.py in starter code
    alpha : float, must be positive
        Defines precision for the prior over the weights
        p(w) = Normal(0, alpha^{-1} I)
    beta : float, must be positive
        Defines precision for the likelihood of t_n given features x_n
        p(t_n | x_n, w) = Normal( w^T x_n, beta^{-1})

    Examples
    --------
    >>> N, D = 100, 1
    >>> prng = np.random.RandomState(0)
    >>> x_ND = prng.randn(N, D)
    >>> x_ND.shape == (N, D)
    True

    >>> t_N = 5 * x_ND[:,0] + 1
    >>> t_N.shape == (N,)
    True

    >>> from FeatureTransformPolynomial import PolynomialFeatureTransform
    >>> txfm = PolynomialFeatureTransform(order=1, input_dim=D)

    >>> alpha = 1.0
    >>> beta = 20.0
    >>> ppe = LinearRegressionPosteriorPredictiveEstimator(txfm, alpha, beta)
    >>> ppe = ppe.fit(x_ND, t_N)
    >>> ppe.mean_M.shape
    (2,)
    >>> ppe.mean_M
    array([0.99964554, 4.99756957])

    ## Check log evidence
    >>> log_ev = ppe.fit_and_calc_log_evidence(x_ND, t_N)
    >>> isinstance(log_ev, float)
    True
    >>> np.round(log_ev, 5)
    37.28976
    """

    def __init__(self, feature_transformer, alpha=1.0, beta=1.0):
        self.feature_transformer = feature_transformer
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.M = self.feature_transformer.get_feature_size() # num features


    def fit(self, x_ND, t_N):
        ''' Fit this estimator to provided training data

        Args
        ----
        x_ND : 2D array, shpae (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim
        t_N : 1D array, shape (N,)
            Outputs

        Returns
        -------
        self. Internal attributes updated.
        '''
        phi_NM = self.feature_transformer.transform(x_ND)
        ## TODO update mean_M and precision_MM via Bishop PRML formulas 
        self.mean_M = np.zeros(self.M)
        self.precision_MM = np.zeros((self.M, self.M))  # S_N_inverse
        I = np.eye(self.M)
        S_N_inverse = self.alpha * I + self.beta * np.matmul(np.transpose(phi_NM), phi_NM)
        S_N = np.linalg.inv(S_N_inverse)
        m_N = np.matmul(self.beta*S_N, np.matmul(np.transpose(phi_NM),t_N))
        self.mean_M = m_N
        self.precision_MM = S_N_inverse
        return self

    def predict(self, x_ND):
        ''' Make predictions of output value for each provided input feature vectors

        Args
        ----
        x_ND : 2D array, shape (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim

        Returns
        -------
        t_est_N : 1D array, size (N,)
            Each entry at index n is prediction given features in row n
            See Bishop PRML Eq. 3.58
        '''
        phi_NM = self.feature_transformer.transform(x_ND)
        N, M = phi_NM.shape
        ## TODO compute mean of predictive
        prediction = np.zeros(N)
        for i in range(N):
            mean = np.matmul(np.transpose(self.mean_M),phi_NM[i])
            #variance = 1/self.beta + np.matmul(np.transpose(phi_NM[i]),np.matmul(np.linalg.inv(self.precision_MM),phi_NM[i]))
            #prediction[i] = np.random.normal(mean,variance,1)
            prediction[i] = mean
        return prediction

    def predict_variance(self, x_ND):
        ''' Make predictions of output variance for each provided input feature vectors

        Args
        ----
        x_ND : 2D array, shape (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim

        Returns
        -------
        var_N : 1D array, size (N,)
            Each entry at index n is the variance given features in row n
            See Bishop PRML Eq. 3.59
        '''
        phi_NM = self.feature_transformer.transform(x_ND)
        N, M = phi_NM.shape
        ## TODO compute variance of predictive
        var_N = np.zeros(N)
        for i in range(N):
            variance = 1/self.beta + np.matmul(np.transpose(phi_NM[i]),np.matmul(np.linalg.inv(self.precision_MM),phi_NM[i])) ##here
            var_N[i] = variance
        return var_N

    def score(self, x_ND, t_N):
        ''' Compute the average log probability of provided dataset

        Assumes we will AVERAGE over the posterior distribution on w computed by this estimator.

        Args
        ----
        x_ND : 2D array, shpae (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim
        t_N : 1D array, shape (N,)
            Outputs

        Returns
        -------
        avg_log_proba : float
        '''
        N = x_ND.shape[0]
        self.fit(x_ND, t_N)
        mean_of_t_N = self.predict(x_ND)
        var_of_t_N = self.predict_variance(x_ND)

        total_log_proba = scipy.stats.norm.logpdf(t_N, mean_of_t_N, np.sqrt(var_of_t_N))
        return np.sum(total_log_proba) / N

    def fit_and_calc_log_evidence(self, x_ND, t_N):
        ''' Compute log evidence for observed data given hyperparameters, marginalizing out weights

        First, performs fit to update posterior given provided data (as required).

        Next, computes log marginal likelihood (log evidence) of the data, a calculation which
        requires the posterior parameters from 'fit'.

        See Bishop PRML textbook Eq. 3.86

        Args
        ----
        x_ND : 2D array, shpae (N, D)
            Each row is a 'raw' feature vector of size D
            D is same as self.feature_transformer.input_dim
        t_N : 1D array, shape (N,)
            Outputs

        Returns
        -------
        avg_log_proba : float
            Log PDF value of observed data

        Examples
        --------
        ## Setup
        >>> from FeatureTransformPolynomial import PolynomialFeatureTransform
        >>> txfm = PolynomialFeatureTransform(order=1, input_dim=1)
        >>> x0 = np.zeros((1,1))

        >>> alpha = 1.0
        >>> beta = 1.0
        >>> ppe = LinearRegressionPosteriorPredictiveEstimator(txfm, alpha, beta)
        >>> log_ev = ppe.fit_and_calc_log_evidence(x0, np.asarray([0.0]))
        >>> np.round(log_ev, 6)
        -1.265512
        '''
        ## Fit posterior to the data
        self.fit(x_ND, t_N)

        ## Transform the raw features to polynomial ones
        phi_NM = self.feature_transformer.transform(x_ND)
        N, M = phi_NM.shape
        
        ## TODO perform evidence calculation
        I = np.eye(self.M)
        det_A = np.linalg.det(self.alpha * I + self.beta * np.matmul(np.transpose(phi_NM), phi_NM))
        E_mN = self.beta/2 * np.linalg.norm(t_N-np.matmul(phi_NM,self.mean_M),ord=2)+self.alpha/2 * np.matmul(np.transpose(self.mean_M),self.mean_M)
        log_ev = M/2 * np.log(self.alpha) + N/2 * np.log(self.beta) - E_mN - 1/2 * np.log(det_A) - N/2 * np.log(2 * math.pi)
        return log_ev



