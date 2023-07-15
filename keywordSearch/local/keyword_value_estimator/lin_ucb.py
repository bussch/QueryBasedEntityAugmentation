import numpy as np


class LinUCBModel(object):
    """Implementation fo the the linUCB model described in (http://proceedings.mlr.press/v15/chu11a/chu11a.pdf).

            Args:
                alpha (float) : determines the amount of exploration. Larger alpha = more emphasis on upper bound.
    """

    def __init__(self, alpha, sender=None):
        self.A_matrix = None
        self.A_inverse = None
        self.b = None
        self.theta = None

        self.alpha = alpha

        self.init_bool = False

    def _initialize_linucb(self, number_of_features):
        """
            Initialize bookkeeping matricies according to amount of features.
        """
        self.A_matrix = np.identity(number_of_features)

        self.init_bool = True

        self.A_inverse = np.identity(number_of_features)
        self.b = np.zeros(number_of_features)

        self.theta = np.matmul(self.A_inverse, self.b)

    def predict(self, featurized_terms):
        """
            Batch score terms using x*Theta + alpha*UCB where,
                x = featurized term vector
                Theta = learned model weights
                alpha = exploration parameter
                UCB = upper confidence bound describing the best (optimistically) upper bound on score we can expect
        """
        if type(self.A_matrix) is not np.ndarray:
            self._initialize_linucb(len(featurized_terms[0]))

        # (x * Theta) + (alpha * sqrt(x^T * A' * x))
        projections = self.A_inverse @ featurized_terms.T
        ucb = np.sqrt((featurized_terms * projections.T).sum(axis=-1))
        scores = (featurized_terms @ self.theta) + (self.alpha * ucb)

        return scores

    def partial_fit(self, sample_x_feat, sample_y):
        """
            Batch update our matricies and vectors using streaming Sherman–Morrison–Woodbury updates.

            Returns:
                error (float) : calculated as MSE
        """

        # Initialize bookkeeping
        if type(self.A_matrix) is not np.ndarray:
            self._initialize_linucb(len(sample_x_feat[0]))

        error = (((sample_x_feat @ self.theta) - sample_y) ** 2).mean()
        delta_f = (sample_y.reshape(-1, 1) * sample_x_feat).sum(axis=0)
        delta_b = np.sum([np.outer(x.T, x) for x in sample_x_feat], axis=0)
        self.A_matrix += delta_b
        self.b += delta_f
        self.A_inverse = np.linalg.inv(self.A_matrix)
        self.theta = np.matmul(self.A_inverse, self.b)

        return error

    def remove_samples(self, sample_x_feat, sample_y):
        """
            Batch removes samples from matricies and vectors using streaming Sherman–Morrison–Woodbury updates.
        """

        delta_f = (sample_y.reshape(-1, 1) * sample_x_feat).sum(axis=0)
        delta_b = np.sum([np.outer(x.T, x) for x in sample_x_feat], axis=0)
        self.A_matrix -= delta_b
        self.b -= delta_f
        self.A_inverse = np.linalg.inv(self.A_matrix)
        self.theta = np.matmul(self.A_inverse, self.b)

    def get_weights(self):
        if self.init_bool:
            return self.theta
        return []