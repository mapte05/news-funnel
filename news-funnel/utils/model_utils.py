import time
import os
import logging
from collections import Counter
from ps2_general_utils import get_minibatches

import numpy as np


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
load word embeddings
'''
def load_embeddings(embedding_file, normalize=lambda token: token.lower()):
    embedding_dimension = None
    for line in open(embedding_file).readlines():
        embedding_dimension = len(line.strip().split()) - 1
        break

    embeddings = []
    id_to_token = []
    token_to_id = {}
    
    # Special start tokens
    special_tokens = ['<START>', '<UNKNOWN>']
    for token in special_tokens:
        token_to_id[token] = len(embeddings)
        id_to_token.append(token)
        embeddings.append(np.random.normal(0, 0.9, (embedding_dimension,)))        
    
    # load the embeddings from a file
    for line in open(embedding_file).readlines():
        sp = line.strip().split()
        sp[0] = normalize(sp[0])
        
        token_to_id[sp[0]] = len(embeddings)
        id_to_token.append(sp[0])
        embeddings.append(np.array([float(x) for x in sp[1:]]))

    return embeddings, token_to_id, id_to_token


    
'''
Load dataset (i.e. dev, test, verification)
'''
def load_and_preprocess_data(data_path, article_file, title_file, embeddings, dataset_type):

    print "Loading", dataset_type, "data...",
    start = time.time()
    with open(os.path.join(data_path, article_file)) as af:
        sentences = list(af.readlines())
    with open(os.path.join(data_path, title_file)) as tf:
        summaries = list(tf.readlines())
    article_set = read_txt(os.path.join(data_path, article_file))
    title_set = read_txt(os.path.join(data_path, title_file))
    print "took {:.2f} seconds".format(time.time() - start)

    print "Vectorizing", dataset_type, "data...",
    start = time.time()
    article_set = None   # TODO turn words into vectors using our embeddings
    title_set = None
    print "took {:.2f} seconds".format(time.time() - start)

    return article_set, title_set


if __name__ == '__main__':
    pass