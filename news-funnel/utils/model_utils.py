import time
import os
import logging
from collections import Counter
from general_utils import get_minibatches

import numpy as np


class Config(object):
    lowercase = True


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
def read_txt(article_file, title_file, lowercase=False, max_example=None):
    with open(article_file) as af:
        sentences = list(af.readlines())
    with open(title_file) as tf:
        summaries = list(tf.readlines())
    return [sentences, summaries]


def get_tokens(data_set):
    result = set()
    sentences, summaries = data_set
    for s in sentences:
        for w in s.strip().split():
            result.add(w)
    for s in summaries:
        for w in s.strip().split():
            result.add(w)
    return result


'''
load word embeddings
'''
def load_and_preprocess_embeddings(data_path, embeddings_file):
    print "Loading pretrained embeddings...",
    start = time.time()

    # load the embeddings from a file
    word_vectors = {}
    for line in open(os.path.join(config.data_path,config.embedding_file)).readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (parser.n_tokens, 50)), dtype='float32')

    tokens = get_tokens(train_set)
    tokens |= get_tokens(dev_set)
    tokens = list(tokens)

    for token in tokens:
        i = parser.tok2id[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]
    print "took {:.2f} seconds".format(time.time() - start)


    return embeddings_matrix

    
'''
Load dataset (i.e. dev, test, verification)
'''
def load_and_preprocess_data(data_path, article_file, title_file, embeddings, dataset_type):
    config = Config()

    print "Loading", dataset_type, "data...",
    start = time.time()
    article_set = read_txt(os.path.join(data_path, article_file),   # read in the data from the files - [sentences, titles]
                           lowercase=config.lowercase)
    title_set = read_txt(os.path.join(data_path, title_file),   # read in the data from the files - [sentences, titles]
                           lowercase=config.lowercase)
    print "took {:.2f} seconds".format(time.time() - start)

    print "Vectorizing", dataset_type, "data...",
    start = time.time()
    article_set = None   # TODO turn words into vectors using our embeddings
    title_set = None
    print "took {:.2f} seconds".format(time.time() - start)

    return article_set, title_set


if __name__ == '__main__':
    pass