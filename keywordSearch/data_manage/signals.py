import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *


class Signal(object):
    def __init__(self, processedToken, originList, isLocal):
        self.keyword = processedToken
        self.originList = originList
        self.isLocal = isLocal
        self.borrowedOriginList = []

    def getTermFrequency(self):
        return sum([origin[2] for origin in self.originList])

    # Just breaks up originList into originTokens, originAttr, originCounts tuples
    def getOriginData(self):
        return zip(*self.originList)

    # Meant to emulate the old way we generated signals--where a signal would appear for each attribute it appeared in
    def splitByAttribute(self):
        totalTF = self.getTermFrequency()
        attrDict = {}  # Create a dictionary from <attribute> -> <origin tokens from attribute>
        for originToken, originAttr, originCount in self.originList:
            if originAttr not in attrDict:
                attrDict[originAttr] = []
            attrDict[originAttr].append((originToken, originAttr, originCount))

        # Split into distinct signals and add a new origin token with (None, None, remainingCount) to maintain
        # TF across tuple (otherwise we would only get TF for this given attribute)
        splitSignalList = [Signal(self.keyword, attrDict[originAttr], self.isLocal) for originAttr in attrDict]
        for newSignal in splitSignalList:
            newSignal.originList.append((None, None, totalTF - newSignal.getTermFrequency()))

        # Recreate as Signals
        returned_signals = [Signal(self.keyword, attrDict[originAttr], self.isLocal) for originAttr in attrDict]
        for newSignal in returned_signals:
            newSignal.borrowedOriginList = self.borrowedOriginList
        return returned_signals

    def equals(self, other):
        return self.keyword == other.keyword and self.originList == other.originList


class FeatureConstructor(object):
    """docstring for FeatureConstructor"""

    def __init__(self):
        super(FeatureConstructor, self).__init__()
        self.originalToken = {}

    def getSignalsOfSingleTuple(self, record, header, isLocal):
        stop_words = set(stopwords.words('english'))

        translator = str.maketrans('', '', string.punctuation.replace('-', ''))
        stemmer = PorterStemmer()

        signalList = []

        originDict = {}
        self.originalToken[record[0]] = {}

        for idx, columnValue in enumerate(record[1:]):
            if header is None:
                headerValue = ''
            else:
                headerValue = header[idx + 1]
            result = columnValue.translate(translator)

            tokenResult = nltk.word_tokenize(result)
            for originalToken in tokenResult:
                if originalToken.lower() not in stop_words:
                    stemmedToken = stemmer.stem(originalToken)
                    if stemmedToken not in originDict:
                        originDict[stemmedToken] = {}
                    originTup = (originalToken, headerValue)
                    if originTup not in originDict[stemmedToken]:
                        originDict[stemmedToken][originTup] = 0
                    originDict[stemmedToken][originTup] += 1

        for stemmedToken in originDict:
            signalList.append(Signal(stemmedToken, [originTup + (originDict[stemmedToken][originTup],) for originTup in
                                                    originDict[stemmedToken]], isLocal))

        return signalList
