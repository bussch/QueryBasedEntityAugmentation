import numpy as np
from local.keyword_value_estimator import lin_ucb
from local.sender_base import SenderBase


class SenderDatasetLevel(SenderBase):
    """Wrapper for the linUCB model.

        Args:
            config['alpha'] (float) : determines the amount of exploration. Larger alpha = more emphasis on upper bound.
    """

    def __init__(self, config, receiver):
        super(SenderDatasetLevel, self).__init__(config, receiver)

        self.local_model = lin_ucb.LinUCBModel(self.config['alpha'], self)
        self.external_model = lin_ucb.LinUCBModel(self.config['alpha'], self)

    def __str__(self):
        return 'dataset_level'

    def generate_query(self, tuple_id, query_length):
        """
            Generate query given current state of model.
        """

        if tuple_id not in self.sent_words:
            self.sent_words[tuple_id] = dict()
            for signal in self.signalIndex[tuple_id]:
                self.sent_words[tuple_id][signal.keyword] = False

        term_list_local = []
        term_list_external = []

        for signal in self.signalIndex[tuple_id]:
            unsupervisedID = str(tuple_id) + "_unsupervised"
            if not signal.isLocal and unsupervisedID in signal.borrowedOriginList:
                if signal.keyword not in self.sent_words[tuple_id]:
                    self.sent_words[tuple_id][signal.keyword] = False
                for splitSignal in signal.splitByAttribute():
                    term_list_external.append(splitSignal)
            else:
                for splitSignal in signal.splitByAttribute():
                    term_list_local.append(splitSignal)

        featurized_terms_external = np.array([self.featurize_external_term(x, tuple_id) for x in term_list_external])
        featurized_terms_local = np.array([self.featurize_term(x, tuple_id) for x in term_list_local])

        scores_local = self.local_model.predict(featurized_terms_local)
        scores_external = []
        if len(featurized_terms_external) > 0:
            scores_external = self.external_model.predict(featurized_terms_external)

        scores = np.append(np.array(scores_local), np.array(scores_external))
        term_list = np.append(np.array(term_list_local), np.array(term_list_external))

        # Select top-n keywords
        if self.config['p_thresh'] is None:
            selected_signals = sorted(list(zip(term_list, scores)), key=(lambda x: x[1]), reverse=True)[:query_length]

        # Select keywords until a given probability mass is hit
        else:
            # Take softmax of ranked_signals
            scores = np.exp(scores) / sum(np.exp(scores))
            ranked_signals = sorted(list(zip(term_list, scores)), key=(lambda x: x[1]), reverse=True)

            mass = 0
            selected_signals = []
            for rank in range(len(ranked_signals)):
                mass += ranked_signals[rank][1]
                selected_signals.append(ranked_signals[rank])
                if mass > self.config['p_thresh'] or len(selected_signals) == query_length:
                    break

        for s in selected_signals:
            self.sent_words[tuple_id][s[0].keyword] = True

        return selected_signals

    def update_model(self, tuple_id, sample_x_signals, sample_y):
        """
            Update state of model.
        """
        sample_external_signals = []
        sample_y_external = []
        sample_local_signals = []
        sample_y_local = []
        for idx, x in enumerate(sample_x_signals):
            unsupervisedID = str(tuple_id) + "_unsupervised"
            if not x.isLocal and unsupervisedID in x.borrowedOriginList:
                sample_external_signals.append(x)
                sample_y_external.append(sample_y[idx])
            else:
                sample_local_signals.append(x)
                sample_y_local.append(sample_y[idx])

        sample_x_feat_local = np.array([self.featurize_term(x, tuple_id) for x in sample_local_signals])
        sample_x_feat_external = np.array([self.featurize_external_term(x, tuple_id) for x in sample_external_signals])

        if (len(sample_x_feat_external) != 0):
            fit = []
            if (len(sample_x_feat_local) != 0):
                fit = self.local_model.partial_fit(sample_x_feat_local, np.array(sample_y_local))
            return np.append(fit, self.external_model.partial_fit(sample_x_feat_external, np.array(sample_y_external)))

        return self.local_model.partial_fit(sample_x_feat_local, np.array(sample_y_local))
