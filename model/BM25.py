#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import math
from six import iteritems
from six.moves import xrange
from collections import defaultdict
from multiprocessing import Pool, Manager
from tqdm import tqdm
import json


# BM25 parameters.
PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):
    def __init__(self, corpus=None, works=1):
        if corpus != None:
            self.corpus_size = 0   # len(corpus)
            self.avgdl = 0.0  # sum(map(lambda x: float(len(x)), corpus)) / self.corpus_size
        else:
            self.corpus_size = 0
            self.avgdl = 0.0

        self.corpus = corpus
        self.works = works
        self.f = []
        self.df = defaultdict(int)
        self.idf = {}
        self.number = []
        self.length = []

        # for word, freq in iteritems(self.df):
        #     self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def initialize(self):
        for document in tqdm(self.corpus):
            frequencies = defaultdict(int)
            patent = json.loads(document)
            self.number.append(patent['patent_number'])
            self.length.append(len(patent['words']))
            self.corpus_size += 1
            for word in patent['words']:
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                self.df[word] += 1

        self.avgdl = float(sum(self.length)) / self.corpus_size
        for word, freq in self.df.items():
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
        # self.parm1 = PARAM_K1 * (1 - PARAM_B + PARAM_B * self.corpus_size / self.avgdl)

    def get_score(self, document, index, average_idf):
        score = 0
        words = set(self.f[index].keys()) & set(document)
        for word in words:
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            f = self.f[index][word]
            score += (idf * f * (PARAM_K1 + 1) / (f + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.length[index] / self.avgdl)))
        return score

    def get_scores(self, documents, average_idf, k=10):
        result = []
        # print(self.corpus_size)
        for document in tqdm(documents):
            scores = []
            for index in range(self.corpus_size):
                score = self.get_score(document, index, average_idf)
                scores.append(score)
            scores = dict(zip(range(len(scores)), scores))
            scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
            result.append([self.number[v[0]] for v in scores])

        return result


def initialize(corpus):
    f = []
    df = defaultdict(int)
    for document in tqdm(corpus):
        frequencies = defaultdict(int)
        for word in document:
            frequencies[word] += 1
        f.append(frequencies)

        for word, freq in iteritems(frequencies):
            df[word] += 1

    return f, df


def get_bm25_weights(corpus, number):
    bm25 = BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())

    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf,number=number)
        weights.append(scores)

    return weights




