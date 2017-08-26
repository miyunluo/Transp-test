# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

import sys
import time
import numpy as np
import pandas as pd

if len(sys.argv) == 1:
    print("ERROR: Please specify implementation to benchmark, 'sknn' or 'nolearn'.")
    sys.exit(-1)

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

from ml_util import Dataset
from sklearn.base import clone
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report
from matplotlib.backends.backend_pdf import PdfPages
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt


read_dataset = Dataset('adult')

X_train, X_test, y_train, y_test = train_test_split(
        read_dataset.num_data, read_dataset.target,
        train_size=0.40,
        )
Mask = X_train["Mask"]
del read_dataset.num_data['Mask']
del X_train['Mask']
del X_test['Mask']

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
#Normalize all training and test data
X_train = pd.DataFrame(scaler.transform(X_train), columns=(read_dataset.num_data.columns))
X_test  = pd.DataFrame(scaler.transform(X_test),  columns=(read_dataset.num_data.columns))


classifiers = []

if 'sknn' in sys.argv:
    from sknn.platform import gpu32
    from sknn.mlp import Classifier, Layer, Convolution

    clf = Classifier(
        layers=[
            # Convolution("Rectifier", channels=10, pool_shape=(2,2), kernel_shape=(3, 3)),
            Layer('Rectifier', units=200),
            Layer('Softmax')],
        learning_rate=0.01,
        learning_rule='nesterov',
        learning_momentum=0.9,
        batch_size=300,
        valid_size=0.0,
        n_stable=10,
        n_iter=20,
        verbose=True)
    classifiers.append(('sknn.mlp', clf))

if 'nolearn' in sys.argv:
    from sknn.platform import gpu32
    from nolearn.lasagne import NeuralNet, BatchIterator
    from lasagne.layers import InputLayer, DenseLayer
    from lasagne.nonlinearities import softmax
    from lasagne.updates import nesterov_momentum

    clf = NeuralNet(
        layers=[
            ('input', InputLayer),
            ('hidden1', DenseLayer),
            ('output', DenseLayer),
            ],
        input_shape=(None, 784),
        output_num_units=10,
        output_nonlinearity=softmax,
        eval_size=0.0,

        more_params=dict(
            hidden1_num_units=200,
        ),

        update=nesterov_momentum,
        update_learning_rate=0.02,
        update_momentum=0.9,
        batch_iterator_train=BatchIterator(batch_size=300),

        max_epochs=10,
        verbose=1)
    classifiers.append(('nolearn.lasagne', clf))


RUNS = 2

for name, orig in classifiers:
    times = []
    accuracies = []
    for i in range(RUNS):
        start = time.time()

        clf = clone(orig)
        clf.random_state = int(time.time())
        clf.fit(X_train, y_train, Mask.as_matrix() )

        accuracies.append(clf.score(X_test, y_test))
        times.append(time.time() - start)

    a_t = np.array(times)
    a_s = np.array(accuracies)

    import pickle
    filename = 'mask_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    y_pred = clf.predict(X_test)

    print("\n"+name)
    print("\tAccuracy: %5.2f%% ±%4.2f" % (100.0 * a_s.mean(), 100.0 * a_s.std()))
    print("\tTimes:    %5.2fs ±%4.2f" % (a_t.mean(), a_t.std()))
    print("\tReport:")
    print(classification_report(y_test, y_pred))

    #-------------------------------------
    #clf = pickle.load(open(filename, 'rb'))
    #y_pred = clf.predict(X_test)

    #print("\n"+name)
    #print("\tAccuracy: %5.2f%% ±%4.2f" % (100.0 * a_s.mean(), 100.0 * a_s.std()))
    #print("\tTimes:    %5.2fs ±%4.2f" % (a_t.mean(), a_t.std()))
    #print("\tReport:")
    #print(classification_report(y_test, y_pred))

def random_intervene(X, cols):
    n = X.shape[0]
    order = np.random.permutation(range(n))
    X_int = np.array(X)

    for c in cols:
        X_int[:,c] = X_int[order, c]
    return X_int


def average_local_influence(dataset, cls, X):
    average_local_inf = {}
    iters = 10
    f_columns = dataset.num_data.columns
    sup_ind = dataset.sup_ind
    #y_pred = cls.predict(X)

    for sf in sup_ind:
        local_influence = np.zeros(y_pred.shape[0])
        ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
        
        for i in xrange(0, iters):
            X_inter = random_intervene(np.array(X), ls)
            y_pred_inter = cls.predict(X_inter)
            local_influence = local_influence+ (y_pred == y_pred_inter)*1.
        average_local_inf[sf] = 1 - (local_influence/iters).mean()
        print(sf, average_local_inf[sf])
        
    return average_local_inf


labelfont = {}
def plot_series(series, xlabel, ylabel):
    plt.figure(figsize=(5,4))
    series.sort_values(inplace=True, ascending=False)
    series.plot(kind="bar")
    plt.xticks(rotation = 45, ha = 'right', size='small')
    plt.xlabel(xlabel, labelfont)
    plt.ylabel(ylabel, labelfont)
    plt.ylim(0, 0.12)
    plt.tight_layout()
    pp = PdfPages('figure'+'.pdf')
    print ('Writing to figure'+ '.pdf')
    pp.savefig(bbox_inches='tight')
    pp.close()
    #plt.show()

def plot_influence():
    print("All_dataset")
    average_local_inf = average_local_influence(read_dataset, clf, X_test)
    average_local_inf_series = pd.Series(average_local_inf, index=average_local_inf.keys())
    f = open('all.txt', 'w')
    f.write(str(average_local_inf_series))
    f.close()
    print("Start plot")
    plot_series(average_local_inf_series, 'Feature', 'QII on Outcomes')

plot_influence()


def random_intervene_point(X, cols, x0):
    """ Randomly intervene on a a set of columns of x from X. """
    n = X.shape[0]
    order = np.random.permutation(range(n))
    X_int = np.tile(x0, (n, 1))
    for c in cols:
        X_int[:, c] = X[order, c]
    return X_int

def unary_individual_influence(dataset, cls, x_ind, X):
    y_pred = cls.predict(x_ind.reshape(1, -1))

    average_local_inf = {}
    iters = 1
    f_columns = dataset.num_data.columns
    sup_ind = dataset.sup_ind
    for sf in sup_ind:
        local_influence = np.zeros(y_pred.shape[0])

        ls = [f_columns.get_loc(f) for f in sup_ind[sf]]
        
        for i in xrange(0, iters):
            X_inter = random_intervene_point(np.array(X), ls, x_ind)
            y_pred_inter = cls.predict(X_inter)
            local_influence = local_influence + (y_pred == y_pred_inter)*1.
        
        average_local_inf[sf] = 1 - (local_influence/iters).mean()
        #print(sf, average_local_inf[sf])
    return average_local_inf

def plot_influence_1():
    print('\n')
    print("Individual")
    f = open('result.txt','w')
    data_dim = read_dataset.num_data.shape[0]
    for i in range(data_dim):
        print("Processing person: ", i)
        x_individual = scaler.transform(read_dataset.num_data.ix[i].reshape(1, -1))
        
        average_local_inf = unary_individual_influence(read_dataset, clf, x_individual, X_test)
        average_local_inf_series = pd.Series(average_local_inf, index=average_local_inf.keys())
        f.write("ix: "+ str(i) +'\n')
        f.write(str(average_local_inf_series))
        f.write('\n')
        f.write('\n')
        #plot_series(average_local_inf_series, i,'Feature', 'QII on Outcomes')
    f.close()

plot_influence_1()
