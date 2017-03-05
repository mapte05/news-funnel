import time
import os
import logging
from collections import Counter
from general_utils import get_minibatches

import numpy as np


class Config(object):
    data_path = './data'
    train_article_file = 'train/valid.article.filter.txt' # TODO: replace at actual training time with 'train/train.article.txt'
    train_title_file = 'train/valid.title.filter.txt' # TODO: replace at actual training time with 'train/train.title.txt'
    dev_article_file = 'train/valid.article.filter.txt'
    dev_title_file = 'train/valid.title.filter.txt'
    test_article_file = 'giga/input.txt' # also need to test on duc2003/duc2004
    test_title_file = 'giga/task1_ref0.txt'

    embedding_file = 'glove.6B.50d.txt' #TODO: replace with 'glove.6B.200d.txt
    embedding_dimension = 50
    
    start_token = '<START>'
    unknown_token = '<UNK>'

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
    return sentences, summaries


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
    

def load_and_preprocess_data():
    config = Config()

    print "Loading data...",
    start = time.time()

    # read in the data from the files - [sentences, titles]
    train_set = read_txt(os.path.join(config.data_path, config.train_article_file), 
                            os.path.join(config.data_path, config.train_title_file),
                           lowercase=config.lowercase)
    dev_set = read_txt(os.path.join(config.data_path, config.dev_article_file),
                        os.path.join(config.data_path, config.dev_title_file), 
                         lowercase=config.lowercase)
    test_set = read_txt(os.path.join(config.data_path, config.test_article_file),
                        os.path.join(config.data_path, config.test_title_file),
                          lowercase=config.lowercase)

    
    tokens |= get_tokens(train_set)
    tokens |= get_tokens(dev_set)
    tokens = [config.start_token, config.unknown_token] + list(tokens)

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