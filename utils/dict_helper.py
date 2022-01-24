from collections import OrderedDict

import torch

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', "''", '...', '``', '-', '--', "'", '"', '<', '>', '/', '..', '=', '+', '»', '~', '«']

class Dict(object):
    def __init__(self, data=None, lower=True):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.lower = lower
        # Special entries will not be pruned.
        self.special = []
        self.idxToBow = {}
        self.bowToIdx = {}

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)

    def size(self):
        return len(self.idxToLabel)

    # Initialize bow vocabulary
    def init_bow_vocab(self, bow_size, stopwords_file=None, freq=0):
        if stopwords_file is not None:
            stopwords = []
            for line in open(stopwords_file):
                word = line.strip()
                stopwords.append(word)
            stopwords.extend(punctuations)

        # Only keep the `size` most frequent entries.
        freq_list = []
        for i in range(len(self.frequencies)):
            if self.frequencies[i] > freq:
                freq_list.append(self.frequencies[i])
        freq = torch.tensor(freq_list)
        _, idx = torch.sort(freq, 0, True)
        idx = idx.tolist()

        for i in idx:
            if len(self.idxToBow) == bow_size:
                break
            if self.idxToLabel[i] in stopwords:
                continue
            j = len(self.idxToBow)
            self.idxToBow[i] = j
            self.bowToIdx[j] = i
        return len(self.idxToBow)
    
    def get_bow_ids(self):
        return [self.bowToIdx[i] for i in range(len(self.idxToBow))]

    # Convert Idxs to BOW vectors
    def convertToBow(self, idx):
        bow = [0] * len(self.idxToBow)
        for i in idx:
            if i in self.idxToBow:
                bow[self.idxToBow[i]] += 1
        return bow
    
    # Write BOW to a file.
    def writeBowFile(self, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            for i in range(len(self.idxToBow)):
                label = self.idxToLabel[self.bowToIdx[i]]
                file.write('%s %d\n' % (label, i))

        file.close()

    # Load entries from a file.
    def loadFile(self, filename):
        for line in open(filename):
            fields = line.split()
            label = fields[0]
            idx = int(fields[1])
            self.add(label, idx)

    # Write entries to a file.
    def writeFile(self, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))

        file.close()

    def loadDict(self, idxToLabel):
        for i in range(len(idxToLabel)):
            label = idxToLabel[i]
            self.add(label, i)

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None, freq=1):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = freq
        else:
            self.frequencies[idx] += freq

        return idx

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, size, freq=0):
        if size > self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq_list = []
        for i in range(len(self.frequencies)):
            if self.frequencies[i] > freq:
                freq_list.append(self.frequencies[i])
        freq = torch.tensor(freq_list)
        freq_list, idx = torch.sort(freq, 0, True)
        freq_list = freq_list.tolist()
        idx = idx.tolist()

        newDict = Dict()
        newDict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for f, i in list(zip(freq_list, idx))[:size]:
            newDict.add(self.idxToLabel[i], freq=f)

        return newDict

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return vec

    def convertToIdxandOOVs(self, labels, unkWord, bosWord=None, eosWord=None, oovs=None):
        vec = []
        if oovs is None:
            oovs = OrderedDict()

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        for label in labels:
            id = self.lookup(label, default=unk)
            if id != unk:
                vec += [id]
            else:
                if label not in oovs:
                    oovs[label] = len(oovs)+self.size()
                oov_num = oovs[label]
                vec += [oov_num]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec), oovs

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.

    def convertToLabels(self, idx, stop, oovs=None):
        labels = []

        for i in idx:
            if i == stop:
                break
            if i < self.size():
                labels += [self.getLabel(i)]
            else:
                labels += [list(oovs.items())[i-self.size()][0]]

        return labels