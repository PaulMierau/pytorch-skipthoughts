import sys
import numpy

import torch
import torch.nn as nn

import torch.nn.functional as F

from collections import OrderedDict

import nltk
from nltk.tokenize import word_tokenize

import logging

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger = logging.getLogger()
logger.addHandler(consoleHandler)
logger.setLevel(logging.NOTSET)




class SkipThoughts(nn.Module):
    def __init__(self, dirStr: str, dictionary: dict, fixedEmb: bool = False, normalized: bool = True):
        super(SkipThoughts, self).__init__()

        self.dirStr = dirStr
        self.fixed_emb = fixedEmb
        self.normalized = normalized
        self.dictionary = dictionary

    def preprocess(self, x):
        X = []
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        for t in x:
            sents = sent_detector.tokenize(t)
            result = ''
            for s in sents:
                tokens = word_tokenize(s)
                result += ' ' + ' '.join(tokens)
            X.append(result)

        wordIdx = [[self.dictionary[word] for word in s.split()] for s in X]

        tensorWordIdx = torch.zeros(len(wordIdx), max([len(i) for i in wordIdx])) # needs numpy base for large batches
        for i in range(len(tensorWordIdx)):
            tensorWordIdx[i,:len(wordIdx[i])] = torch.tensor(wordIdx[i], dtype=torch.int64)

        return tensorWordIdx.long()

    def loadEmbedding(self, dictionary: dict,  filePath: str):
        logging.info(f"Loading table: {filePath}")
        embedding = nn.Embedding(num_embeddings=len(self.dictionary) + 1,
                                      embedding_dim=620,
                                      padding_idx=0,
                                      sparse=False)

        parameters = numpy.load(filePath, encoding='latin1', allow_pickle=True)
        weights = torch.zeros(len(dictionary) + 1, 620)
        for i in range(len(weights) - 1):
            weights[i + 1] = torch.from_numpy(parameters[i])
        embedding.load_state_dict({'weight': weights})
        return embedding


class UniSkipThoughts(SkipThoughts):
    def __init__(self, dirStr: str, dictionary: dict, dropout: float = 0, fixedEmb: bool = False, normalized: bool = True):
        super(UniSkipThoughts, self).__init__(dirStr, dictionary, fixedEmb, normalized)
        self.dropout = dropout

        self.embedding = self.loadEmbedding(self.dictionary, dirStr + '/utable.npy')

        if fixedEmb:
            self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(input_size=620,
                          hidden_size=2400,
                          batch_first=True,
                          dropout=self.dropout)
        self.loadModel(dirStr + "/uni_skip.npz")

    def selectResult(self, x, lengths):
        X = torch.zeros(x.size(0), 2400)
        for i in range(len(x)):
            X[i] = x[i][lengths[i]-1]
        return X

    def loadModel(self, modelPath: str):
        logging.info(f"Loading model: {modelPath}")
        params = numpy.load(modelPath, encoding='latin1', allow_pickle=True)
        states = OrderedDict()
        states['bias_ih_l0'] = torch.zeros(7200)
        states['bias_hh_l0'] = torch.zeros(7200)
        states['weight_ih_l0'] = torch.zeros(7200, 620)
        states['weight_hh_l0'] = torch.zeros(7200, 2400)
        states['weight_ih_l0'][:4800] = torch.from_numpy(params['encoder_W']).t()
        states['weight_ih_l0'][4800:] = torch.from_numpy(params['encoder_Wx']).t()
        states['bias_ih_l0'][:4800] = torch.from_numpy(params['encoder_b'])
        states['bias_ih_l0'][4800:] = torch.from_numpy(params['encoder_bx'])
        states['weight_hh_l0'][:4800] = torch.from_numpy(params['encoder_U']).t()
        states['weight_hh_l0'][4800:] = torch.from_numpy(params['encoder_Ux']).t()
        self.gru.load_state_dict(states)

    def forward(self, input):
        lengths = [len(s.split(' ')) for s in input]
        input = self.preprocess(input)
        x = self.embedding(input)
        y, hn = self.gru(x)
        y = self.selectResult(y, lengths)
        if self.normalized:
            y = torch.nn.functional.normalize(y)
        return y


class BiSkipThoughts(SkipThoughts):

    def __init__(self, dirStr: str, dictionary: dict, dropout: float = 0, fixedEmb: bool = False, normalized: bool = True):
        super(BiSkipThoughts, self).__init__(dirStr, dictionary, fixedEmb, normalized)
        self.dropout = dropout

        self.embedding = self.loadEmbedding(self.dictionary, dirStr + '/btable.npy')

        if fixedEmb:
            self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(input_size=620,
                          hidden_size=1200,
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=True)

        self.loadModel(dirStr + "/bi_skip.npz")


    def loadModel(self, modelPath: str):
        logging.info(f"Loading model: {modelPath}")
        params = numpy.load(modelPath, encoding='latin1', allow_pickle=True)
        states = OrderedDict()
        states['bias_ih_l0'] = torch.zeros(3600)
        states['bias_hh_l0'] = torch.zeros(3600)  # must stay equal to 0
        states['weight_ih_l0'] = torch.zeros(3600, 620)
        states['weight_hh_l0'] = torch.zeros(3600, 1200)

        states['bias_ih_l0_reverse'] = torch.zeros(3600)
        states['bias_hh_l0_reverse'] = torch.zeros(3600)  # must stay equal to 0
        states['weight_ih_l0_reverse'] = torch.zeros(3600, 620)
        states['weight_hh_l0_reverse'] = torch.zeros(3600, 1200)

        states['weight_ih_l0'][:2400] = torch.from_numpy(params['encoder_W']).t()
        states['weight_ih_l0'][2400:] = torch.from_numpy(params['encoder_Wx']).t()
        states['bias_ih_l0'][:2400] = torch.from_numpy(params['encoder_b'])
        states['bias_ih_l0'][2400:] = torch.from_numpy(params['encoder_bx'])
        states['weight_hh_l0'][:2400] = torch.from_numpy(params['encoder_U']).t()
        states['weight_hh_l0'][2400:] = torch.from_numpy(params['encoder_Ux']).t()

        states['weight_ih_l0_reverse'][:2400] = torch.from_numpy(params['encoder_r_W']).t()
        states['weight_ih_l0_reverse'][2400:] = torch.from_numpy(params['encoder_r_Wx']).t()
        states['bias_ih_l0_reverse'][:2400] = torch.from_numpy(params['encoder_r_b'])
        states['bias_ih_l0_reverse'][2400:] = torch.from_numpy(params['encoder_r_bx'])
        states['weight_hh_l0_reverse'][:2400] = torch.from_numpy(params['encoder_r_U']).t()
        states['weight_hh_l0_reverse'][2400:] = torch.from_numpy(params['encoder_r_Ux']).t()
        self.gru.load_state_dict(states)

    def forward(self, input):
        lengths = [len(s.split(' ')) for s in input]

        x = self.preprocess(input)
        x = self.embedding(x)

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        y, hn = self.gru(x)

        hn = hn.transpose(0, 1).contiguous()
        hn = hn.view(len(input), 2 * hn.size(2))

        if self.normalized:
            hn = torch.nn.functional.normalize(hn)

        return hn


class Encoder(object):
    def __init__(self, dirStr: str, dropout: float = 0, fixedEmb: bool = False, normalized: bool = True):
        self.dirStr = dirStr
        self.dropout = dropout
        self.fixedEmb = fixedEmb
        self.normalized = normalized
        self.dictionary = self.loadDictionary(dirStr)
        self.uniSkip = UniSkipThoughts(dirStr, self.dictionary, dropout, fixedEmb, normalized)
        self.biSkip = BiSkipThoughts(dirStr, self.dictionary, dropout, fixedEmb, normalized)

    def loadDictionary(self, dirStr: str):
        logging.info("Loading dictionary")
        with open(dirStr + '/dictionary.txt', 'r', encoding="utf8") as file:
            words = file.readlines()

        dictionary = {}
        for idx, word in enumerate(words):
            dictionary[word.strip()] = idx + 1
        return dictionary

    def encode(self, input: list):
        uFeatures = self.uniSkip(input)
        bFeatures = self.biSkip(input)
        return torch.cat([uFeatures, bFeatures], 1)

if __name__ == '__main__':

    dirStr = 'models'

    encoder = Encoder(dirStr)

    test = ["Hey, how are you?", "This sentence is a lie"]

    result = encoder.encode(test)

    print(result)




