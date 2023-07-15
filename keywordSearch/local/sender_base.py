import random
import copy

from data_manage.datastore import Datastore
from local.keyword_value_estimator import featurization_utils
import pickle as pickle
import os
from additionalScripts.output_config import OutputConfig


class SenderBase(object):
    """Base class for all sender models.

        Any new model should subclass this class to help guarantee that it will work with run_experiments.py.

        Args:
            config (dictionary): dictionary of <setting: value> pairs. Can freely add new settings here that may only
                be used by specific models.
            receiver (Receiver): the external receiver. Used to initialize external features.
    """

    def __init__(self, config, receiver, dataset=None):
        super(SenderBase, self).__init__()
        self.config = config

        # Process data
        if dataset is None:
            dataset = Datastore(self.config['data_file_path'])
        self.lin_attribute_headers = dataset.header[1:]  # Drop ID attribute
        self.idf = None
        self.signalIndex = None
        self._idf_buckets = None
        self._len_buckets = None
        self._normTF_buckets = None
        self._processData(dataset)

        if (self.config['distribution'] == 'zipfian'):
            self._zipf_prob = None
            self._zipf_events = None

        # Bias data
        self._external_data = None
        self._external_signalIndex = None
        self._mixed_data = None
        self._maximumTF_ext = dict()

        # External features
        self._has_matched = dict()
        self._max_external_tf = dict()
        self._external_tf = dict()
        self._max_external_tf_unsupervised = dict()
        self._external_tf_unsupervised = dict()
        self._tf_buckets_external = None

        if self.config['external_feat_specific'] or self.config['unsupervised_term_borrowing']:
            self._process_ext_features(receiver)

        # Unsupervised term borrowing instance variables
        self.tried_unsupervised = dict()
        self.unsupervised_matched = dict()
        self.rr_zero = dict()
        self.sent_words = dict()
        self.FEATURES_EXTERNAL = []

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        shallow_copyable = ['config', 'lin_attribute_headers', 'idf', 'signalIndex', '_idf_buckets', '_len_buckets',
                            '_normTF_buckets', '_zipf_prob', '_zipf_events', '_has_matched',
                            '_max_external_tf', '_external_tf', '_tf_buckets_external']

        for k, v in self.__dict__.items():
            if k in shallow_copyable:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        return result

    # def __getstate__(self):
    #     """Remove large data structures prior to pickling. NOTE: Windows systems will SPAWN processes when Process is
    #         is called. This has the effect of also calling __getstate__, which means deleting these structures on
    #         spawning. Consider simply commenting this override (function) out if running on a Windows system and
    #         pickling the sender object.
    #     """
    #     state = self.__dict__.copy()
    #     del state['signalIndex']
    #     del state['idf']
    #     return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _save_obj(self, obj, name):
        """Convenience method for saving objects."""
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def _load_obj(self, name):
        """Convenience method for loading objects."""
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def _processData(self, data):
        """Either loads in preprocessed data structures and buckets or uses the passed in Data object to create them
            and then store them for this experiment and future experiments.

            Data Structures:
                idf: normalized IDF for each stemmed term
                signalIndex: dictionary of <tuple_id: <signal_list>>
                _x_buckets: equidepth (as possible) buckets for x feature for one-hot encoding
        """

        RECORD_DIR_PATH = OutputConfig(self.config['config_path']).paths['processed_data']

        print(RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/")
        if not os.path.exists(RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/"):
            # Create data structures

            self.idf, self.signalIndex = data.process(isLocal=True)

            os.makedirs(RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/")
            self._save_obj(self.idf, RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/idf")
            self._save_obj(self.signalIndex,
                           RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/signalIndex")
        else:
            # Load existing data structures
            self.idf = self._load_obj(RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/idf")
            self.signalIndex = self._load_obj(
                RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/signalIndex")

        print(RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/buckets/")
        if not os.path.exists(RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/buckets/"):
            # Create buckets
            self._idf_buckets = featurization_utils.findBucket(self.config, 7, list(self.idf.values()))
            self._len_buckets = featurization_utils.findBucket(self.config, 6, [len(signal.keyword) for signal_list in
                                                                                self.signalIndex.values() for signal in
                                                                                signal_list])
            maximumTF_local = {key: max([signal.getTermFrequency() for signal in self.signalIndex[key]])
                               for key in self.signalIndex}
            normalizedTFCount = [signal.getTermFrequency() / maximumTF_local[key]
                                 for key in maximumTF_local for signal in self.signalIndex[key]]
            self._normTF_buckets = featurization_utils.findBucket(self.config, 6, normalizedTFCount)

            os.makedirs(RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/buckets/")
            self._save_obj(self._idf_buckets,
                           RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/buckets/idf_buckets")
            self._save_obj(self._len_buckets,
                           RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/buckets/len_buckets")
            self._save_obj(self._normTF_buckets,
                           RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/buckets/normTT_buckets")
        else:
            # Load existing buckets
            self._idf_buckets = self._load_obj(
                RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/buckets/idf_buckets")
            self._len_buckets = self._load_obj(
                RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/buckets/len_buckets")
            self._normTF_buckets = self._load_obj(
                RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/buckets/normTT_buckets")

    def generateZipfianDistribution(self, seed):
        """Generate a zipfian distribution for tuple sampling. Rankings are shuffled so every
        call results in different probabilities for each tuple.
        """
        print('Generating random zipfian distribution with s=1')
        self._zipf_events = [id for id in self.signalIndex]
        random.seed(seed)
        random.shuffle(self._zipf_events)
        random.seed()
        self._zipf_prob = [1 / i for i in range(1, len(self.signalIndex) + 1)]
        self._zipf_prob = [x / sum(self._zipf_prob) for x in self._zipf_prob]

    def pickTupleToJoin(self):
        """Randomly select a tuple_id from signalIndex.

            Returns:
                A string representing the uniformly selected tuple_id for the sender's dataset
        """
        if self.config['distribution'] == 'uniform':
            return random.choice(list(self.signalIndex.keys()))
        elif self.config['distribution'] == 'zipfian':
            chance = random.uniform(0, 1)
            cumulative = 0
            i = -1
            while cumulative < chance:
                i += 1
                cumulative += self._zipf_prob[i]
            return self._zipf_events[i]
        elif (self.config['distribution'] == 'fixed') or (self.config['distribution'] == 'uniform_fixed'):
            return self.tuple_series.pop(0)
        else:
            print('{} is not a valid intent selection distribution'.format(self.config['distribution']))
            exit()

    def featurize_term(self, signal, source_tupleID, external_feat_specific=False):
        """Returns a featurized representation of a signal.

            Args:
                signal (Signal) : the signal to featurize.
                source_tupleID (string) : the tuple_id of the tuple where this signal came from.
                tuple_specific_features (boolean) : whether tuple-specific features should be included or not
                    (if not, then tuple-specific features will be 0).

            Returns:
                A list of floats representing the featurized version of this signal from this tuple
        """
        if self.config['external_feat_specific'] or self.config['external_feat_specific_unsupervised']:
            external_feat_specific = True
        return featurization_utils.get_characteristics(self, signal, source_tupleID, external_feat_specific)

    def featurize_external_term(self, signal, source_tupleID, external_feat_specific=False):
        """Returns a featurized representation of a signal.

            Args:
                signal (Signal) : the signal to featurize.
                source_tupleID (string) : the tuple_id of the tuple where this signal came from.
                tuple_specific_features (boolean) : whether tuple-specific features should be included or not
                    (if not, then tuple-specific features will be 0).

            Returns:
                A list of floats representing the featurized version of this signal from this tuple
        """
        if self.config['external_feat_specific'] or self.config['external_feat_specific_unsupervised']:
            external_feat_specific = True

        return featurization_utils.get_characteristics_external(self, signal, source_tupleID, external_feat_specific)

    def _process_ext_features(self, receiver):
        """Either loads in preprocessed external buckets or uses the passed in Receiver object to create them
            and then store them for this experiment and future experiments.

            Args:
                receiver (Receiver) : the Receiver from which the external features will be derived.

            Data Structures:
                _x_buckets_external : equidepth (as possible) buckets for x external feature for one-hot encoding
        """

        RECORD_DIR_PATH = OutputConfig(self.config['config_path']).paths['processed_data']

        self._external_signalIndex = receiver.signalIndex

        print(RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/ext_buckets/")
        if not os.path.exists(RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/ext_buckets/"):
            # Create data structures
            maximumTF_external = {key: max([signal.getTermFrequency() for signal in receiver.signalIndex[key]])
                                  for key in receiver.signalIndex}
            normalizedTFCount = [signal.getTermFrequency() / maximumTF_external[key]
                                 for key in maximumTF_external for signal in receiver.signalIndex[key]]
            self._tf_buckets_external = featurization_utils.findBucket(6, normalizedTFCount)

            os.makedirs(RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/ext_buckets/")
            self._save_obj(self._tf_buckets_external,
                           RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/ext_buckets/tf_bucket")
        else:
            # Load existing data structures
            self._tf_buckets_external = self._load_obj(
                RECORD_DIR_PATH + self.config['dataset_name'] + "_sender/ext_buckets/tf_bucket")

    def update_tuple_specific_features(self, local_tuple_id, external_tuple_ids, external_signal_index):
        """used to update tuple-specific features as new matches are found.

            Args:
                local_tuple_id (string) : the ID of the tuple to update tuple-specific features for
                external_tuple_ids (list) : a list of string IDs for external tuples for which tuple-specific features
                    should be derived. Expected to be the list of external tuple IDs that match with this local tuple
                    specified by local_tuple_id.
                external_signal_index (dict) : the signalIndex dictionary of the Receiver. Used to access the external
                    signals associated with the tuple IDS in external_tuple_ids.

            Data Structures Updated:
                _has_matched : keeps track of <local_tuple_id: <external_tuple_id: True/False>> to prevent redundant work.
                if 'external_feat_specific' is True:
                    _max_external_tf (dict) : the maximum external term frequency (count of the term that appears most often)
                        for a external tuple <external_tuple_id: max_TF>. Used for normalization.
                    _external_tf (dict) : mapping of <local_tuple_id: <stemmed_term: <external_tuple_id: TF>>>. A single
                        stemmed_term may appear in multiple matches, so we keep track of each in order to average them.
                if 'supervised_term_borrowing' is True:
                    signalIndex : adds external signals to the signalIndex
        """

        # Unsupervised term borrowing
        exhausted_all_terms = False
        if self.config['unsupervised_term_borrowing']:
            if local_tuple_id not in self.rr_zero:
                self.rr_zero[local_tuple_id] = 1
            else:
                self.rr_zero[local_tuple_id] += 1
            num_sent = 0.0
            total_sent = 0.0
            for w in self.sent_words[local_tuple_id]:
                total_sent += 1.0
                if self.sent_words[local_tuple_id][w] == True:
                    num_sent += 1.0
            if (num_sent / total_sent) >= 0.7:
                exhausted_all_terms = True

        for external_tuple_id in external_tuple_ids:
            if self.config['unsupervised_term_borrowing']:
                if exhausted_all_terms:
                    if (local_tuple_id not in self.unsupervised_matched) or (
                            local_tuple_id in self.unsupervised_matched and external_tuple_id not in
                            self.unsupervised_matched[local_tuple_id]):
                        self.tried_unsupervised[local_tuple_id] = False
                        local_term_list = [signal.keyword for signal in self.signalIndex[local_tuple_id]]
                        for external_signal in external_signal_index[external_tuple_id]:
                            if external_signal.keyword not in local_term_list:
                                unsupervisedID = str(local_tuple_id) + "_unsupervised"
                                external_signal.borrowedOriginList.append(unsupervisedID)
                                self.signalIndex[local_tuple_id].append(external_signal)
                        if local_tuple_id not in self.unsupervised_matched:
                            self.unsupervised_matched[local_tuple_id] = list()
                        self.unsupervised_matched[local_tuple_id].append(external_tuple_id)
                    # External unsupervised term frequency
                    if self.config['external_feat_specific_unsupervised']:
                        if local_tuple_id not in self._max_external_tf_unsupervised:
                            self._max_external_tf_unsupervised[local_tuple_id] = dict()
                            self._external_tf_unsupervised[local_tuple_id] = dict()
                        if external_tuple_id not in self._max_external_tf_unsupervised[local_tuple_id]:
                            self._max_external_tf_unsupervised[local_tuple_id][external_tuple_id] = \
                                max([signal.getTermFrequency() for signal in external_signal_index[external_tuple_id]])
                            for external_signal in external_signal_index[external_tuple_id]:
                                if external_signal.keyword not in self._external_tf_unsupervised[local_tuple_id]:
                                    self._external_tf_unsupervised[local_tuple_id][external_signal.keyword] = dict()
                                self._external_tf_unsupervised[local_tuple_id][external_signal.keyword][
                                    external_tuple_id] = external_signal.getTermFrequency()

            # skip work if we have already seen these matches
            if local_tuple_id in self._has_matched and \
                    external_tuple_id in self._has_matched[local_tuple_id]:
                continue

            # External term frequency
            if self.config['external_feat_specific']:
                if local_tuple_id not in self._max_external_tf:
                    self._max_external_tf[local_tuple_id] = dict()
                    self._external_tf[local_tuple_id] = dict()
                if external_tuple_id not in self._max_external_tf[local_tuple_id]:
                    self._max_external_tf[local_tuple_id][external_tuple_id] = \
                        max([signal.getTermFrequency() for signal in external_signal_index[external_tuple_id]])
                    for external_signal in external_signal_index[external_tuple_id]:
                        if external_signal.keyword not in self._external_tf[local_tuple_id]:
                            self._external_tf[local_tuple_id][external_signal.keyword] = dict()
                        self._external_tf[local_tuple_id][external_signal.keyword][
                            external_tuple_id] = external_signal.getTermFrequency()

            # Term borrowing
            if self.config['supervised_term_borrowing']:
                local_term_list = [signal.keyword for signal in self.signalIndex[local_tuple_id]]

                for external_signal in external_signal_index[external_tuple_id]:
                    new_borr = []
                    unsupervisedID = str(local_tuple_id) + "_unsupervised"
                    for borrID in external_signal.borrowedOriginList:
                        if unsupervisedID != borrID:
                            new_borr.append(borrID)
                    external_signal.borrowedOriginList = new_borr
                    if local_tuple_id not in external_signal.borrowedOriginList:
                        external_signal.borrowedOriginList.append(local_tuple_id)
                    if external_signal.keyword not in local_term_list:
                        self.signalIndex[local_tuple_id].append(external_signal)

            # Register match
            if (not self.config['unsupervised_term_borrowing']):
                if local_tuple_id not in self._has_matched:
                    self._has_matched[local_tuple_id] = {external_tuple_id: True}
                else:
                    self._has_matched[local_tuple_id][external_tuple_id] = True

        return []

    def generate_query(self, tuple_id, query_length):
        """Must be implemented by child class. Generates a list of signals given the current model.

            Args:
                tuple_id (string) : ID of tuple to generate a query for. Used for featurization.
                query_length (int) : amount of keywords to include in query.

            Returns:
                list of Signal objects representing the generated query.
        """
        raise NotImplementedError("generate_query not implemented by child class")

    def update_model(self, tuple_id, sample_x_signals, sample_y):
        """Must be implemented by child class. Updates the current model given a list of signals.

            Args:
                tuple_id (string) : ID of tuple the sample_x_signals were derived from. Used for featurization.
                sample_x_signals (list) : list of Signals for which to update the model.
                sample_y (list) : response variables. List of rewards (float) for each Signal in sample_x_signals.

            Returns:
                loss/error (float) : the loss/error of the model on the (x,y) pairs prior to updating.
        """
        raise NotImplementedError("update_model not implemented by child class")

    def model_is_fitted(self):
        """Indicates whether a model is able to distinguish between signals. Some models may give every signal a score
            of 0 until they receive positive feedback.

            Returns:
                  boolean indicating whether the model has been fitted to some examples.
        """
        raise NotImplementedError("update_model not implemented by child class")

    def get_weights(self):
        raise NotImplementedError("update_model not implemented by child class")
