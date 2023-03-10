from nltk.corpus import wordnet as wn
import pandas as pd

def findBucket(size, data):
    buckets = []
    categorical = pd.qcut(data, size, duplicates='drop')
    bin_sizes = categorical.value_counts()
    bins = [i for i in pd.qcut(data, size, duplicates='drop').categories]
    for i in range(len(bins)):
        buckets.append((bins[i].left, bins[i].right, bin_sizes[i]))
    return buckets


def get_characteristics(self, signal, tupleID, ts_features):
    word = signal.keyword
    originTokens, originAttr, originCounts = signal.getOriginData()
    self.FEATURES = []
    X = []

    #
    #   Local-specific features
    #
        # Local idf of term
    idf = 0
    if signal.isLocal:
        idf = self.idf[word]
    for bucket in self._idf_buckets:
        self.FEATURES.append('{} < IDF <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf <= bucket[1]:  # 0.0786094674556213
            X.append(1)
        else:
            X.append(0)

        # Term Frequency features
    if not hasattr(self, 'maximumTF'):
        self.maximumTF = {}
    if signal.isLocal and tupleID not in self.maximumTF:
        self.maximumTF[tupleID] = max([signal.getTermFrequency() for signal in self.signalIndex[tupleID]])

            # TF for whole tuple
    self.FEATURES.append('maximum_TF_normalized')
    normTF = 0
    if signal.isLocal:
        normTF = (signal.getTermFrequency() / self.maximumTF[tupleID])
    X.append(normTF)
    for bucket in self._normTF_buckets:
        self.FEATURES.append('{} < normTF <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < normTF <= bucket[1]:
            X.append(1)
        else:
            X.append(0)
            # TF among attributes
    tfAttrFeatures = [0] * len(self.lin_attribute_headers)
    for tfAttrIndex, header in enumerate(self.lin_attribute_headers):
        self.FEATURES.append('TF_{}_normalized'.format(header))
        if signal.isLocal:
            for attrIndex, attr in enumerate(originAttr):
               if header == attr:
                    tfAttrFeatures[tfAttrIndex] += originCounts[attrIndex]
    X += [total / sum(originCounts) for total in tfAttrFeatures]

        # Local attributes
    for feat in self.lin_attribute_headers:
        self.FEATURES.append(feat)
        if signal.isLocal and feat in originAttr:
            X.append(1)
        else:
            X.append(0)

    #
    #   Keyword features
    #
        # Binned keyword length
    for bucket in self._len_buckets:
        self.FEATURES.append('{} < len <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < len(word) <= bucket[1]:
            X.append(1)
        else:
            X.append(0)

        # Has a non-alpha character
    self.FEATURES.append('non-alpha 1+')  # 0.1393491317848441
    encode = 0
    for c in word:
        if not c.isalpha():
            encode = 1
            break
    X.append(encode)

        # Original token features
    isTitle, isUpper, noun, verb, adjective, adverb = 0, 0, 0, 0, 0, 0
    for term in originTokens:
        if term == None:
            continue

        # Is Title-cased
        if term.istitle():
            isTitle = 1

        # Is Uppercased
        if term.isupper():
            isUpper = 1

        # Word Type
        word_morph = wn.morphy(term.lower())
        if word_morph != None:
            noun += len(wn.synsets(word_morph, wn.NOUN))
            verb += len(wn.synsets(word_morph, wn.VERB))
            adjective += len(wn.synsets(word_morph, wn.ADJ))
            adverb += len(wn.synsets(word_morph, wn.ADV))

    self.FEATURES.append('Is title-cased')
    X.append(isTitle)

    self.FEATURES.append('Is upper-cased')
    X.append(isUpper)

    total = max(1, noun + verb + adjective + adverb)
    self.FEATURES.append('Noun')
    X.append(noun / total)
    self.FEATURES.append('Verb')
    X.append(verb / total)
    self.FEATURES.append('Adjective')
    X.append(adjective / total)
    self.FEATURES.append('Adverb')
    X.append(adverb / total)

    #
    #   External-specific features (general)
    #
        # Is borrowed term
    if self.config['supervised_term_borrowing']:
        self.FEATURES.append('Is borrowed term')
        if not signal.isLocal and tupleID in signal.borrowedOriginList:
            X.append(1)
        else:
            X.append(0)

    #
    #   External-specific features (tuple-specific)
    #
    if self.config['external_feat_specific'] or self.config['external_feat_specific_unsupervised']:
            # Term Frequency whole tuple
        average_normTF_ext = 0
        self.FEATURES.append('TF_external_normalized')
        if ts_features and tupleID in self._external_tf and word in self._external_tf[tupleID]:
            average_normTF_ext = sum([self._external_tf[tupleID][word][extID]/self._max_external_tf[tupleID][extID]
                                   for extID in self._external_tf[tupleID][word]]) / len(self._max_external_tf[tupleID])
            if average_normTF_ext>1:
                print(signal.isLocal, average_normTF_ext)
            X.append(average_normTF_ext)
        else:
            X.append(0)

    #
    #   Unsupervised-Term-Borrowing-specific features (tuple-specific)
    #
    if self.config['external_feat_specific_unsupervised']:
            # Term Frequency whole tuple
        average_normTF_ext = 0
        self.FEATURES.append('unsupervised_TF_external_normalized')
        if ts_features and tupleID in self._external_tf_unsupervised and word in self._external_tf_unsupervised[tupleID]:
            average_normTF_ext = sum([self._external_tf_unsupervised[tupleID][word][extID] / self._max_external_tf_unsupervised[tupleID][extID]
                                      for extID in self._external_tf_unsupervised[tupleID][word]]) / len(self._max_external_tf_unsupervised[tupleID])
            if average_normTF_ext>1:
                print(signal.isLocal, average_normTF_ext)
            X.append(average_normTF_ext)
        else:
            X.append(0)
    return X


def get_characteristics_external(self, signal, tupleID, ts_features):
    word = signal.keyword
    self.FEATURES_EXTERNAL = []
    X = []

    #
    #   Local-specific features
    #
    # Local idf of term
    idf = 0
    if word in self.idf:
        idf = self.idf[word]
    for bucket in self._idf_buckets:
        self.FEATURES_EXTERNAL.append('{} < LOCAL IDF <= {}'.format(bucket[0], bucket[1]))
        if bucket[0] < idf <= bucket[1]:  # 0.0786094674556213
            X.append(1)
        else:
            X.append(0)

    #
    #   Unsupervised-Term-Borrowing-specific features (tuple-specific)
    #
    if self.config['external_feat_specific_unsupervised']:
        # Term Frequency whole tuple
        average_normTF_ext = 0
        self.FEATURES_EXTERNAL.append('unsupervised_TF_external_normalized')
        if ts_features and tupleID in self._external_tf_unsupervised and word in self._external_tf_unsupervised[
            tupleID]:
            average_normTF_ext = sum([self._external_tf_unsupervised[tupleID][word][extID] /
                                      self._max_external_tf_unsupervised[tupleID][extID]
                                      for extID in self._external_tf_unsupervised[tupleID][word]]) / len(
                self._max_external_tf_unsupervised[tupleID])
            if average_normTF_ext > 1:
                print(signal.isLocal, average_normTF_ext)
            X.append(average_normTF_ext)
        else:
            X.append(0)

    return X