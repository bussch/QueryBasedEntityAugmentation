import csv

class Oracle(object):
    """docstring for Oracle"""

    def __init__(self, filename):
        super(Oracle, self).__init__()
        self.data1 = dict()
        self.data2 = dict()

        with open(filename, encoding="latin-1") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            self.header = next(spamreader)

            for row in spamreader:
                if row[0] not in self.data1:
                    self.data1[row[0]] = list()
                    self.data1[row[0]].append(row[1])
                else:
                    self.data1[row[0]].append(row[1])

                if row[1] not in self.data2:
                    self.data2[row[1]] = list()
                    self.data2[row[1]].append(row[0])
                else:
                    self.data2[row[1]].append(row[0])

    def getTotalTrue(self, concept1):
        if concept1 in self.data1:
            return len(self.data1[concept1])

        if concept1 in self.data2:
            return len(self.data2[concept1])

        return 0

    def getRealMatch(self, concept1):
        return self.data1[concept1]

    # Returns all source IDs that have a match in the oracle
    def getAllSourceMatches(self):
        return list(self.data1.keys())

    def getAllLocalMatches(self):
        return list(self.data2.keys())

    def isMatch(self, concept1, concept2):
        if concept1 in self.data1:
            for con in self.data1[concept1]:
                if con == concept2:
                    return True

        if concept1 in self.data2:
            for con in self.data2[concept1]:
                if con == concept2:
                    return True

        return False
