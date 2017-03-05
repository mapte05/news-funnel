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
        
    token_to_id_fn = lambda token: token_to_id[normalize(token)] if normalize(token) in token_to_id else token_to_id['<UNKNOWN>']
    return embeddings, token_to_id_fn, id_to_token

    
'''
Load dataset (i.e. dev, test, verification)
'''
def load_data(article_file):
    articles = []
    with open(article_file) as af:
        for article in af.readlines():
            articles.append(article.split())
    return articles

def preprocess_data(articles, token_to_id, article_length):
    articles = np.array([[token_to_id(word) for word in sentence] for sentence in articles], ndmin=article_length)
    if articles.shape[1] >= article_length:
        articles = articles[:, 0:article_length]
    return articles
    

if __name__ == '__main__':
    pass
