import time
import os
import logging
from collections import Counter
from general_utils import get_minibatches

import numpy as np


class Config(object):
    data_path = './data'
    train_file = 'train.conll'  # TODO replace .conll with the file type that Rush uses
    dev_file = 'dev.conll'  # TODO replace .conll with the file type that Rush uses
    test_file = 'test.conll'    # TODO replace .conll with the file type that Rush uses
    embedding_file = './data/en-cw.txt'


'''
Taken verbatim from PS2
'''
def minibatches(data, batch_size):
    x = np.array([d[0] for d in data])
    y = np.array([d[2] for d in data])
    one_hot = np.zeros((y.size, 3))
    one_hot[np.arange(y.size), y] = 1
    return get_minibatches([x, one_hot], batch_size)

'''
Reads in the text files and outputs Python lists
'''
def read_txt():
    pass


'''
Load
- dataset created by Rush and create dev and test sets
- word embeddings
'''
def load_and_preprocess_data():
    config = Config()

    print "Loading data...",
    start = time.time()

    # read in the data from the files
    train_set = read_txt(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)
    dev_set = read_txt(os.path.join(config.data_path, config.dev_file),
                         lowercase=config.lowercase)
    test_set = read_txt(os.path.join(config.data_path, config.test_file),
                          lowercase=config.lowercase)
    
    print "took {:.2f} seconds".format(time.time() - start)

    print "Loading pretrained embeddings...",
    start = time.time()

    # load the embeddings from a file
    word_vectors = {}
    for line in open(config.embedding_file).readlines():
        pass # TODO this is where we read the lines out of the word embeddings file

    print "took {:.2f} seconds".format(time.time() - start)

    print "Vectorizing data...",
    start = time.time()

    # TODO turn words into vectors using our emeddings
    train_set = None
    dev_set = None
    test_set = None
    print "took {:.2f} seconds".format(time.time() - start)

    print "Preprocessing training data..."
    train_examples = train_set # TODO do something with train_set to make the training examples

    return embeddings_matrix, train_examples, dev_set, test_set,

if __name__ == '__main__':
    pass