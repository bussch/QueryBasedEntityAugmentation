import numpy as np
from local.keyword_value_estimator import lin_ucb
from local.sender_base import SenderBase

class SenderHybrid(SenderBase):
    """Wrapper for the linUCB model. Starts with a single linUCB model for every tuple and then spins up new linUCB
        models for individual tuples if all the following criteria are met:
            - a tuple-specific model hasn't been triggered yet
            - this_tuple_last_rr <= rr_threshold (historically, this tuple has done poorly)
            - last_window_mrr >= current_window_mrr (no positive change in dataset level performance)

        Note that tuple-specific reciprocal rank will only be tracked after the first window_size updates. This is meant
        to avoid spinning up tuple-specific models for tuples that did poorly early on simply due to a lack of feedback.

        Args:
            config['alpha'] (float) : determines the amount of exploration for each model.
                Larger alpha = more emphasis on upper bound.
            config['window_size'] (int) : window size for sliding MRR calculations.
            self.config['rr_threshold'] (float) : RR threshold that demarcates poor performance (less than threshold).
    """
    def __init__(self, config, receiver):
        super(SenderHybrid, self).__init__(config, receiver)
        self.dataset_model = lin_ucb.LinUCBModel(self.config['alpha'], self)

        # Bookkeeping for specific_model triggering events
        self._last_window = None
        self._current_window = None
        self._rr_list = []

        self._examples_seen = dict()
        self._last_tuple_rr = dict()

        # Bookkeeping for tuple-specific models
        self._tuple_models = dict()

    def __str__(self):
        return 'hybrid'

    def generate_query(self, tuple_id, query_length):
        """
            Generate query given current state of the models. Will switch to tuple-specific models if triggering
                conditions are met.
        """
        term_list = [splitSignal for signal in self.signalIndex[tuple_id] for splitSignal in
                         signal.splitByAttribute()]

        # Check triggering event... For this tuple,
        if (tuple_id not in self._tuple_models                                   # 1. We don't already have a specifc model for it
                and tuple_id in self._last_tuple_rr                              # 2. We have a record of trying this tuple before
                and self._last_tuple_rr[tuple_id] <= self.config['rr_threshold'] # 3. Historically, this tuple has done poorly
                and self._last_window is not None
                and self._last_window >= self._current_window):                  # 4. There has been no overall performance increase in
                                                                                 #    dataset_level model over the previous window

            self._tuple_models[tuple_id] = lin_ucb.LinUCBModel(self.config['alpha'])

            x_sample_feat = np.array([self.featurize_term(x, tuple_id, self.config['external_feat_specific']) for x_list, _ in self._examples_seen[tuple_id] for x in x_list])
            y_sample = np.concatenate([y_list for _, y_list in self._examples_seen[tuple_id]])
            loss = self._tuple_models[tuple_id].partial_fit(x_sample_feat, y_sample)

            self.dataset_model.remove_samples(x_sample_feat, y_sample)

        use_ts_features = tuple_id in self._tuple_models and self.config['external_feat_specific']
        featurized_terms = np.array([self.featurize_term(x, tuple_id, use_ts_features) for x in term_list])

        if tuple_id in self._tuple_models:
            scores = self._tuple_models[tuple_id].predict(featurized_terms)
        else:
            scores = self.dataset_model.predict(featurized_terms)

        return sorted(zip(term_list, scores), key=(lambda x: x[1]), reverse=True)[:query_length]

    def update_model(self, tuple_id, sample_x_signals, sample_y):
        """
            Update state of model and RR windows. If a tuple-specific model has been triggered for this tuple_id,
                then that model will be updated.
        """
        reciprocal_rank = max(sample_y)

        # Update RR windows
        self._rr_list.append(reciprocal_rank)
        if len(self._rr_list) == self.config['window_size']:
            self._last_window = self._current_window
            self._current_window = sum(self._rr_list) / len(self._rr_list)
            self._rr_list = []

        # Track tuple-specific RR if the model has trained enough
        if self._current_window is not None:
            self._last_tuple_rr[tuple_id] = reciprocal_rank

        if tuple_id not in self._tuple_models:
            if tuple_id not in self._examples_seen:
                self._examples_seen[tuple_id] = []
            self._examples_seen[tuple_id].append((sample_x_signals, sample_y))

        # Update model
        use_ts_features = tuple_id in self._tuple_models and self.config['external_feat_specific']
        sample_x_feat = np.array([self.featurize_term(x, tuple_id, use_ts_features) for x in sample_x_signals])
        if tuple_id in self._tuple_models:
            return self._tuple_models[tuple_id].partial_fit(sample_x_feat, sample_y)
        else:
            return self.dataset_model.partial_fit(sample_x_feat, sample_y)
