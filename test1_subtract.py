from __future__ import print_function
import logging
import sklearn
import sklearn.ensemble
import sklearn.metrics
import re
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing as preprocessing

logging.basicConfig()

categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)
#print("Train set dimension: ",train_vectors.shape)
#print("Test set dimension: ", test_vectors.shape)
#dict = vectorizer.vocabulary_
#print("Test find value from key GOD: ", dict['GOD'])
#print("Test find key form value 0: ", dict.keys()[dict.values().index(0)])

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)
pred = rf.predict(test_vectors)
print("Accurate number: (over 717) ",(pred==newsgroups_test.target).sum())

a = sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')
print("f1_score: ", a)

c = make_pipeline(vectorizer, rf)
idx = 83
test_instance = newsgroups_test.data[idx]
d = c.predict_proba([newsgroups_test.data[idx]])
print("Predict (pipeline.predict_proba): ", d)
print("Predict (rf.predict_proba): ", rf.predict_proba(test_vectors[idx]))
print("Predict label: ",pred[idx])
print("True label: ", newsgroups_test.target[idx])
print(pred.shape, test_vectors[idx].shape)

class IndexedString(object):
    """String with various indexes."""
    def __init__(self, raw_string, split_expression=r'\W+', bow=True):  
        self.raw = raw_string
        self.as_list = re.split(r'(%s)|$' % split_expression, self.raw)
        self.as_np = np.array(self.as_list)
        non_word = re.compile(r'(%s)|$' % split_expression).match
        self.string_start = np.hstack(
            ([0], np.cumsum([len(x) for x in self.as_np[:-1]])))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, word in enumerate(self.as_np):
            if word in non_vocab:
                continue
            if non_word(word):
                non_vocab.add(word)
                continue
            if bow:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = vocab[word]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) ocurrences"""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, words_to_remove):
        """Returns a string after removing the appropriate words.

        If self.bow is false, replaces word with UNKWORDZ instead of removing
        it.

        Args:
            words_to_remove: list of ids (ints) to remove

        Returns:
            original raw string with appropriate words removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return ''.join([self.as_list[i] if mask[i]
                            else 'UNKWORDZ' for i in range(mask.shape[0])])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in words]))
        else:
            return self.positions[words]


indexed_string = IndexedString(test_instance, bow = True, split_expression=r'\W+')
res = {}

for i in range(len(indexed_string.inverse_vocab)):
    inverse_data = []
    #print(indexed_string.word(i))
    inverse_data.append(indexed_string.inverse_removing([i]))
    result = c.predict_proba(inverse_data)
    #print(result[0][0]-d[0][0])
    res[indexed_string.word(i)] = abs(result[0][0]-d[0][0])
    
t = sorted(res.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
print(t)
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['Host']] = 0
tmp[0,vectorizer.vocabulary_['Posting']] = 0
print("Original : ", d)
print("After removing Distribution and Re: ",rf.predict_proba(tmp))
