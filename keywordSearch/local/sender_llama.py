import random

from data_manage.datastore import Datastore
from local.keyword_value_estimator import llama
from local.sender_base import SenderBase

class SenderLlama(SenderBase):
    def __init__(self, config, receiver):
        self.dataset = Datastore(config['data_file_path'])
        super(SenderLlama, self).__init__(config, receiver, dataset=self.dataset)

        dummy_id = [x for x in self.signalIndex.keys()][0]
        dummy_signal = self.signalIndex[dummy_id][0]
        char_size = len(self.featurize_term(dummy_signal, dummy_id))
        self.model = llama.LinkLlama(char_size, self.config['buffer_size'], self.config['buffer_sample_size'])

    def __str__(self):
        return 'llama'

    def generate_query(self, tuple_id, query_length):
        """Get top query_length keywords ranked by the model and select via e-greedy"""
        exploit_list = self.model.predict_raw_terms(self, tuple_id, self.dataset)
        exploit_list = sorted(exploit_list, key=(lambda x: x[1]), reverse=True)

        terms_to_send = self.select_epsilon_greedy(exploit_list, query_length)

        return terms_to_send

    def update_model(self, tuple_id, sample_x_signals, sample_y):
        self.model.add_sample_to_buffer(sample_x_signals, sample_y)
        return self.model.partial_fit(self.config['split_sample'])

    def select_epsilon_greedy(self, exploit_list, howMany):
        terms_to_send = []
        while len(terms_to_send) < howMany and 0 < len(exploit_list):
            chance = random.uniform(0, 1)
            if chance > self.config['epsilon']:  # Exploit
                terms_to_send.append(exploit_list.pop(0))
            else:  # Explore
                random_term = random.choice(exploit_list)
                terms_to_send.append(random_term)
                exploit_list.remove(random_term)

        return terms_to_send

    def model_is_fitted(self):
        return True