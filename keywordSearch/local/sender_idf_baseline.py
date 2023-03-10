from local.sender_base import SenderBase

class SenderIDFBaseline(SenderBase):
    """IDF Baseline. Sends the top-k terms with the highest IDF.

        Args:
            Takes no additional arguments
    """
    def __init__(self, config, receiver):
        super(SenderIDFBaseline, self).__init__(config, receiver)

    def __str__(self):
        return 'idf_baseline'

    def generate_query(self, tuple_id, query_length):
        """
            Generate query given IDF of terms in entity
        """
        term_list = [splitSignal for signal in self.signalIndex[tuple_id] for splitSignal in
                         signal.splitByAttribute()]

        scores = [self.idf[signal.keyword] for signal in term_list]

        return sorted(zip(term_list, scores), key=(lambda x: x[1]), reverse=True)[:query_length]

    def update_model(self, tuple_id, sample_x_signals, sample_y):
        """
            Do nothing. IDF Baseline is a static policy
        """
        return 0

    def _process_ext_features(self, receiver):
        """
            Do nothing. Technique not applicable to IDF Baseline
        """
        return

    def update_tuple_specific_features(self, local_tuple_id, external_tuple_ids, external_signal_index):
        """
            Do nothing. Technique not applicable to IDF Baseline
        """
        return