import csv
import math

from data_manage.signals import FeatureConstructor

class Datastore():
	"""docstring for Data"""

	def __init__(self, filename):
		super(Datastore, self).__init__()
		csv.field_size_limit(9999999)
		self.filename = None
		self.header = None
		self.table = dict()
		self.filename = filename

		# Read in CSV data
		with open(self.filename, encoding="latin-1") as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',')
			self.header = next(spamreader)
			for row in spamreader:
				self.table[row[0]] = row

	def getRow(self, rowID):
		return str(self.table[rowID])

	def getListRow(self, rowID):
		return self.table[rowID]

	def getConcepts(self):
		stringList = list()
		signalList = list(self.table.values())
		for sig in signalList:
			stringList.append(str(sig))
		return stringList

	def getValues(self):
		return list(self.table.values())

	def getHeader(self):
		return self.header

	def process(self, isLocal=True):
		featureConst = FeatureConstructor()
		countKeywordInDocs = dict()
		listOfTuples = self.getValues()

		idf = dict()
		signalIndex = dict()

		for tuple in listOfTuples:
			tuple_id = tuple[0]

			signals = featureConst.getSignalsOfSingleTuple(tuple, self.getHeader(), isLocal=isLocal)
			for signal in signals:
				if tuple_id not in signalIndex:
					signalIndex[tuple_id] = list()
				signalIndex[tuple_id].append(signal)

				if signal.keyword not in countKeywordInDocs:
					countKeywordInDocs[signal.keyword] = 1
				else:
					countKeywordInDocs[signal.keyword] += 1

		# Calculate final IDF scores
		numDocuments = len(listOfTuples)
		for keyword in countKeywordInDocs:
			idf[keyword] = math.log(numDocuments / float(countKeywordInDocs[keyword]))
		maxIDF = max(list(idf.values()))
		for keyword in idf:
			idf[keyword] = idf[keyword] / maxIDF

		return idf, signalIndex
