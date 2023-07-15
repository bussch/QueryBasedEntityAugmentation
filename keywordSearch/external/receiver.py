import os
import pickle as pickle
from data_manage.datastore import Datastore
from external.keyword_search import KeywordSearch
from additionalScripts.output_config import OutputConfig

class Receiver(object):
	"""docstring for ReceiverCharm"""
	def __init__(self, config):
		super(Receiver, self).__init__()
		self.config = config
		self.signalIndex = dict()
		self._processData(Datastore(self.config['data_file_path']))

		self.returnedTuples = list()
		self.receivedSignals = list()

		self.database = KeywordSearch(self.config['data_file_path'], self.config['dataset_name'], self.config['config_path'])

	def save_obj(self, obj, name):
		with open(name + '.pkl', 'wb') as f:
			pickle.dump(obj, f)

	def load_obj(self, name):
		with open(name + '.pkl', 'rb') as f:
			return pickle.load(f)

	def _processData(self, data):

		RECORD_DIR_PATH = OutputConfig(self.config['config_path']).paths['processed_data']

		if not os.path.exists(RECORD_DIR_PATH + self.config['dataset_name'] + "/"):

			self.idf, self.signalIndex = data.process(isLocal=False)

			os.makedirs(RECORD_DIR_PATH + self.config['dataset_name'] + "/")
			self.save_obj(self.idf, RECORD_DIR_PATH + self.config['dataset_name'] +"/idf")
			self.save_obj(self.signalIndex, RECORD_DIR_PATH + self.config['dataset_name'] + "/signalIndex")
		else:
			self.idf = self.load_obj(RECORD_DIR_PATH + self.config['dataset_name'] +"/idf")
			self.signalIndex = self.load_obj(RECORD_DIR_PATH + self.config['dataset_name'] + "/signalIndex")

	def returnTuples(self, querySignals, top_k):

		self.receivedSignals = querySignals
		returnedIDs = self.database.search(
			' '.join([keyword for signal in querySignals for keyword in signal[0].keyword.split('-')]), top_k)

		if len(returnedIDs) == 0:
			return []

		return returnedIDs
