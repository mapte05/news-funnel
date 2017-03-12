'''
correct use

training: python model.py train
testing: python model.py test param_file

'''
import tensorflow as tf
import numpy as np
import time
import threading
import pickle
import sys
import os
import glob
from utils.model_utils import load_embeddings, load_data, preprocess_data


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    #n_features = 36
    vocab_size = None # set during preprocessing
    context_size = 5 # taken from Rush (C)
    summary_length = None # set during preprocessing
    article_length = None # set during preprocessing
    embed_size = None # set during preprocessing (Rush: D = 200)
    hidden_size = 400 # taken from Rush (H)
    batch_size = 512 # Rush uses 64
    n_epochs = 15 # taken from Rush
    #n_layers = 3 # taken from Rush (L)
    lr = 0.005 # taken from Rush
    lr_decay_base = .5 # taken from Rush
    lr_decay_after_steps = 100000
    lr_staircase = False
    smoothing_window = 2 # taken from Rush (Q)
    beam_size = 5
    encoding_method = "attention" # "attention" or "bag-of-words"
    param_save_step = 1000
    
    max_vocab = 75000 # Nallapati 150k
    max_train_articles = None
    
    # Limits for memory conserve
    max_summary_length = 12 #26
    max_article_length = 24 #96
    
    start_token = None # set during preprocessing
    end_token = None # set during preprocessing
    null_token = None # set during preprocessing
    unknown_token = None # set during preprocessing

    num_batches_for_testing = 1

    saver_path = 'variables/news-funnel-model'
    train_article_file = './data/train/train.article.txt'
    train_title_file = './data/train/train.title.txt'
    preprocessed_articles_file="preprocessed_articles_file.npy"
    preprocessed_summaries_file="preprocessed_summaries_file.npy"

    train_loss_file = "eval/train_losses"
    test_loss_file_root = "eval/test_losses"
    test_results_file_root = "eval/giga_system"
    
    #train_article_file = './data/train/valid.article.filter.txt' # for debug
    #train_title_file = './data/train/valid.title.filter.txt' # for debug
    #preprocessed_articles_file="preprocessed_articles_file_valid.npy"
    #preprocessed_summaries_file="preprocessed_summaries_file_valid.npy"
    
    dev_article_file = './data/train/valid.article.filter.txt'
    dev_title_file = './data/train/valid.title.filter.txt'
    test_article_file = './data/giga/input.txt' # also need to test on duc2003/duc2004
    embedding_file = './data/glove.6B.200d.txt' #TODO: replace with 'glove.6B.200d.txt
    

class RushModel:

    def __init__(self, word2vec_embeddings, config):
        self.word2vec_embeddings = word2vec_embeddings
        self.config = config
        self.defined = False

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, [None, self.config.article_length])
        self.summaries_placeholder = tf.placeholder(tf.int32, [None, self.config.summary_length])
        # self.dropout_placeholder = tf.placeholder(tf.float32, shape=())

    def create_feed_dict(self, inputs_batch, summaries_batch=None):
        """

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {}
        feed_dict[self.input_placeholder] = inputs_batch
        if summaries_batch is not None:
            feed_dict[self.summaries_placeholder] = summaries_batch
        return feed_dict
    
    def do_prediction_step(self, input, context, suppress_unknown=False):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.constant_initializer(0.0)
        embed_init = self.word2vec_embeddings

        with tf.variable_scope("prediction_step", reuse=self.defined):
            output_embeddings = tf.get_variable("E", initializer=embed_init)
            input_embeddings = tf.get_variable("F", initializer=embed_init)
            encoding_embeddings = tf.get_variable("G", initializer=embed_init)
            
            embedded_input = tf.nn.embedding_lookup(ids=input, params=input_embeddings)
            embedded_context = tf.reshape(tf.nn.embedding_lookup(ids=context, params=output_embeddings), (-1, self.config.context_size*self.config.embed_size))
            embedded_context_for_encoding = tf.reshape(tf.nn.embedding_lookup(ids=context, params=encoding_embeddings), (-1, self.config.context_size*self.config.embed_size))
            
            U = tf.get_variable("U", shape=(self.config.context_size*self.config.embed_size, self.config.hidden_size), initializer=xavier_init)
            b1 = tf.get_variable("b1", shape=(1, self.config.hidden_size), initializer=zero_init)
            
            V = tf.get_variable("V",  shape=(self.config.hidden_size, self.config.vocab_size), initializer=xavier_init)
            W = tf.get_variable("W", shape=(self.config.embed_size, self.config.vocab_size), initializer=xavier_init) # TODO: Might need tweaking depend on encoding method
            b2 = tf.get_variable("b2", shape=(1, self.config.vocab_size), initializer=zero_init)

            P = tf.get_variable("P", shape=(self.config.embed_size, self.config.embed_size*self.config.context_size), initializer=xavier_init)
            self.defined = True
        
            if self.config.encoding_method == "bag-of-words":
                encoded = tf.reduce_mean(embedded_context_for_encoding, axis=-2) # average along input
            elif self.config.encoding_method == "attention":
                p = tf.nn.softmax(tf.einsum('ij,bwi,bj->bw', P, embedded_input, embedded_context_for_encoding))
                
                # Smoothing
                start_embedding = tf.nn.embedding_lookup(ids=self.config.start_token, params=encoding_embeddings)
                end_embedding = tf.nn.embedding_lookup(ids=self.config.end_token, params=encoding_embeddings)
                padded_input = tf.concat_v2([
                    tf.tile(tf.expand_dims(tf.expand_dims(start_embedding, 0), 0), [self.config.batch_size, self.config.smoothing_window, 1]),
                    embedded_input,
                    tf.tile(tf.expand_dims(tf.expand_dims(end_embedding, 0), 0), [self.config.batch_size, self.config.smoothing_window, 1])
                ], 1)
                smoothed_input = tf.zeros_like(embedded_input)
                for i in xrange(2*self.config.smoothing_window + 1):
                    smoothed_input += tf.slice(padded_input, [0, i, 0], [-1, self.config.article_length, -1])
                smoothed_input /= 2.*self.config.smoothing_window + 1.
                
                encoded = tf.einsum('bw,bwi->bi', p, smoothed_input)
            else:
                raise Exception("encoding method invalid")
            
            if suppress_unknown:
                b2 = b2 - tf.one_hot([self.config.unknown_token], self.config.vocab_size, on_value=100000.)
            
            h = tf.tanh(tf.matmul(embedded_context, U) + b1)
            logits = tf.matmul(h, V) + tf.matmul(encoded, W) + b2
        
            return logits

    def add_loss_op(self, articles, summaries):
        logits = []
        padded_context = tf.concat_v2([
            tf.fill([self.config.batch_size, self.config.context_size], self.config.start_token), 
            summaries], 1)
        for i in xrange(self.config.summary_length):
            context = tf.slice(padded_context, [0, i], [-1, self.config.context_size])
            logits.append(self.do_prediction_step(articles, context))
        logits = tf.stack(logits, axis=1)
        
        null_mask = tf.not_equal(summaries, self.config.null_token)
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=summaries)
        return tf.reduce_mean(tf.boolean_mask(cross_entropy_loss, null_mask))
        
    def predict(self, articles, method="greedy"):
        if method == "greedy":
            padded_predictions = tf.fill([self.config.batch_size, self.config.context_size], self.config.start_token)
            for i in range(self.config.summary_length):
                context = tf.slice(padded_predictions, [0, i], [-1, self.config.context_size])
                logits = self.do_prediction_step(articles, context, suppress_unknown=True)
                
                # Experiment: use only common words
                #logits = tf.slice(logits, [0, 0], [-1, 30000])
                
                padded_predictions = tf.concat_v2([padded_predictions, tf.expand_dims(tf.to_int32(tf.argmax(logits, axis=1)), -1)], 1)
            return tf.slice(padded_predictions, [0, self.config.context_size], [-1, -1])
        """
        elif method == "beam":
            padded_predictions = tf.fill(self.config.start_token, [self.config.batch_size, self.config.beam_size, self.config.context_size])
            prediction_log_probs = tf.fill(0, [self.config.batch_size, self.config.beam_size])
            for i in range(self.config.summary_length):
                context = tf.slice(padded_predictions, [0, 0, i], [-1, -1, self.config.context_size])
                
                log_probs = prediction_log_probs + tf.nn.log_softmax(logits=do_prediction_step(articles, context)) 
                assert log_probs.get_shape() == (self.config.batch_size, self.config.beam_size, self.config.vocab_size)
            
                best_log_probs, best_words = tf.nn.top_k(input=log_probs, k=self.config.beam_size) 
                assert best_log_probs.get_shape() == (self.config.batch_size, self.config.beam_size, self.config.beam_size)
                assert best_words.get_shape() == (self.config.batch_size, self.config.beam_size, self.config.beam_size)
                
                best_log_probs = tf.reshape(best_log_probs, (self.config.batch_size, self.config.beam_size**2))
                best_words = tf.reshape(best_words, (self.config.batch_size, self.config.beam_size**2))
                prediction_log_probs, best_indices = tf.nn.top_k(input=best_log_probs, k=self.config.beam_size) 
                assert prediction_log_probs.get_shape() == (self.config.batch_size, self.config.beam_size)
                assert best_indices.get_shape() == (self.config.batch_size, self.config.beam_size)
                
                best_beams = tf.mod(best_indices, self.config.beam_size)
                best_subbeams = tf.truncatediv(best_indices, self.config.beam_size)
                
                
                
                # dimensions: batch, beam, beam
                
                
                
                padded_predictions = tf.concat_v2(
                    padded_prediction[best_indices.remove_last_dim + :], 
                    best_indices.last_dim
                )
                # dimensions: 
                
                prediction_log_probs = best_log_probs
                
                padded_predictions = tf.stack()
                prediction_logits = 
        """        
        # else:
        #     raise Exception("predict method not greedy or beam")

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.config.lr, global_step, self.config.lr_decay_after_steps, self.config.lr_decay_base, staircase=self.config.lr_staircase)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        training_op = optimizer.minimize(loss, global_step=global_step)
        grads, vars = zip(*optimizer.compute_gradients(loss))

        return training_op, tf.global_norm(grads), learning_rate

def write_config(config, config_file):
    with open(config_file, 'wb') as outf:
        pickle.dump(config, outf, pickle.HIGHEST_PROTOCOL)

def load_config(config_file):
    with open(config_file, 'rb') as inpf:
        config = pickle.load(inpf)
    return config


def train_main(config_file="config/config_file", debug=True, run_dev=False, reload_data=False): 
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()
    train_articles = None
    train_summaries = None
    dev_articles = None
    dev_summaries = None
    
    if debug:
        config.max_train_articles = 1000

    print "Loading embedding data...",
    start = time.time()
    embeddings, token_to_id, id_to_token = load_embeddings(config.embedding_file, config.max_vocab)
    config.vocab_size = embeddings.shape[0]
    config.embed_size = embeddings.shape[1]
    config.start_token = token_to_id('<s>')
    config.end_token = token_to_id('<e>')
    config.null_token = token_to_id('<null>')
    config.unknown_token = token_to_id('<unk>')
    print "loaded {0} embeddings".format(config.vocab_size)
    print "took {:.2f} seconds".format(time.time() - start)

    print "Loading training data...",
    start = time.time()
    if not reload_data and os.path.isfile(config.preprocessed_articles_file) and os.path.isfile(config.preprocessed_summaries_file):
        train_articles = np.load(config.preprocessed_articles_file)
        train_summaries = np.load(config.preprocessed_summaries_file)
        
        config.article_length = train_articles.shape[1]
        config.summary_length = train_summaries.shape[1]
    else:
        train_articles = load_data(config.train_article_file, config.max_train_articles)
        config.article_length = article_length = min(max(len(x) for x in train_articles) + 1, config.max_article_length)
        train_articles = preprocess_data(train_articles, token_to_id, article_length)
        
        train_summaries = load_data(config.train_title_file, config.max_train_articles)
        config.summary_length = summary_length = min(max(len(x) for x in train_summaries) + 1, config.max_summary_length)
        train_summaries = preprocess_data(train_summaries, token_to_id, summary_length)

        if not debug:
        	np.save(config.preprocessed_articles_file, train_articles)
	        np.save(config.preprocessed_summaries_file, train_summaries)
    assert train_articles.shape[0] == train_summaries.shape[0]
    print "loaded {0} articles, {1} summaries".format(train_articles.shape[0], train_summaries.shape[0])
    print "took {:.2f} seconds".format(time.time() - start)

    print "Loading dev data...",
    start = time.time()
    dev_articles = load_data(config.dev_article_file, config.num_batches_for_testing * config.batch_size)
    dev_articles = preprocess_data(dev_articles, token_to_id, config.article_length)
    
    dev_summaries = load_data(config.dev_title_file, config.num_batches_for_testing * config.batch_size)
    dev_summaries = preprocess_data(dev_summaries, token_to_id, config.summary_length)
    print "took {:.2f} seconds".format(time.time() - start)


    print "writing Config to file"
    write_config(config, config_file)

    def load_train_example(sess, enqueue, coord):
        while True:
            for i in xrange(train_articles.shape[0]):
                sess.run(enqueue, feed_dict={train_article_input: train_articles[i], train_summary_input: train_summaries[i]})
                if coord.should_stop():
                    return

    def load_dev_example(sess, enqueue, coord):
        while True:
            for i in xrange(config.num_batches_for_testing * config.batch_size):
                sess.run(enqueue, feed_dict={dev_article_input: dev_articles[i], dev_summary_input: dev_summaries[i]})
                if coord.should_stop():
                    return

    model = RushModel(embeddings, config)
    
    # Define training pipeline
    train_article_input = tf.placeholder(tf.int32, shape=(config.article_length,))
    train_summary_input = tf.placeholder(tf.int32, shape=(config.summary_length,))
    train_queue = tf.RandomShuffleQueue(1024, 128, [tf.int32, tf.int32], shapes=[(config.article_length,), (config.summary_length,)])
    train_enqueue = train_queue.enqueue([train_article_input, train_summary_input])
    
    train_article_batch, train_summary_batch = train_queue.dequeue_many(config.batch_size)
    train_article_batch = tf.reshape(train_article_batch, (config.batch_size, config.article_length)) # hacky
    train_summary_batch = tf.reshape(train_summary_batch, (config.batch_size, config.summary_length))
    
    train_loss_op = model.add_loss_op(train_article_batch, train_summary_batch)
    training_op, grad_norm_op, lr_op = model.add_training_op(train_loss_op)

    # Define testing pipeline
    dev_article_input = tf.placeholder(tf.int32, shape=(config.article_length,))
    dev_summary_input = tf.placeholder(tf.int32, shape=(config.summary_length,))
    dev_queue = tf.FIFOQueue(1024, [tf.int32, tf.int32], shapes=[(config.article_length,), (config.summary_length,)])
    dev_enqueue = dev_queue.enqueue([dev_article_input, dev_summary_input])
    
    dev_article_batch, dev_summary_batch = dev_queue.dequeue_many(config.batch_size)
    dev_article_batch = tf.reshape(dev_article_batch, (config.batch_size, config.article_length)) # hacky
    dev_summary_batch = tf.reshape(dev_summary_batch, (config.batch_size, config.summary_length))
    
    dev_loss_op = model.add_loss_op(dev_article_batch, dev_summary_batch)
    predictions = model.predict(dev_article_batch)

    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    loss = None


    def test_lite(sess, count):
        testf = open(config.test_results_file_root+count, 'w+')
        tlossf = open(config.test_loss_file_root+count, 'w+')
        loss_sum = 0.
        for i in xrange(config.num_batches_for_testing):
            summaries, loss = sess.run([predictions, dev_loss_op])
            loss_sum += loss
            for summary in summaries.tolist():
                line = []
                for id in summary:
                    if id == config.end_token:
                        break
                    line.append(id_to_token[id])
                i += 1
                testf.write(' '.join(word for word in word_list)+'\n')
        mean_loss = loss_sum / config.num_batches_for_testing
        grad_norm, lr = sess.run([grad_norm_op, lr_op])
        tlossf.write(','.join([mean_loss, grad_norm, lr]) + '\n')


    lf = open(config.train_loss_file, 'w+')
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = [
            threading.Thread(target=load_train_example, args=(sess, train_enqueue, coord)),
            threading.Thread(target=load_dev_example, args=(sess, dev_enqueue, coord))
        ]
        for thread in threads:
            thread.start()
        tf.train.start_queue_runners(sess=sess)

        print 80 * "="
        print "TRAINING"
        print 80 * "="
        counter = 0
        with coord.stop_on_exception():
            start = time.time()
            while True:
                counter += 1
                if counter % 10 == 0:
                    print "10 minibatches took {:.2f} seconds".format(time.time() - start)

                if counter % config.param_save_step == 0:
                    saver.save(sess, config.saver_path, global_step=counter)
                    test_lite(sess, counter/config.save_step)
                    print "SAVED AND TESTED ON PARAMETERS | loss:", loss, "| counter:", counter
                loss, _ = sess.run([loss_op, training_op])
                lf.write(loss+'\n')
                



def test_main(param_file, config_file="config/config_file", load_config_from_file=True, debug=False):
    print >> sys.stderr,  80 * "="
    print >> sys.stderr, "INITIALIZING"
    print >> sys.stderr, 80 * "="
    
    assert len(glob.glob(param_file + "*")) > 0
    
    config = None
    if load_config_from_file:
        config = load_config(config_file)
    else:
        config = Config()

    print >> sys.stderr, "Loading embedding data...",
    start = time.time()
    embeddings, token_to_id, id_to_token = load_embeddings(config.embedding_file, config.max_vocab)
    assert len(embeddings) == config.vocab_size
    print >> sys.stderr,  "took {:.2f} seconds".format(time.time() - start)

    print >> sys.stderr, "Loading test data...",
    start = time.time()
    test_articles = load_data(config.test_article_file)
    test_articles = preprocess_data(test_articles, token_to_id, config.article_length)
    print >> sys.stderr, "took {:.2f} seconds".format(time.time() - start)
    
    def load_example(sess, enqueue, coord):
        for i in xrange(test_articles.shape[0]):
            sess.run(enqueue, feed_dict={article_input: test_articles[i]})
            if coord.should_stop():
                return
        while i % config.batch_size != 0:
            sess.run(enqueue, feed_dict={article_input: test_articles[0]})
            i += 1
            if coord.should_stop():
                return
    
    model = RushModel(embeddings, config)

    article_input = tf.placeholder(tf.int32, shape=(config.article_length,))
    queue = tf.FIFOQueue(1024, [tf.int32], shapes=[(config.article_length,)])
    enqueue = queue.enqueue([article_input])
    
    article_batch = queue.dequeue_many(config.batch_size)
    article_batch = tf.reshape(article_batch, (config.batch_size, config.article_length)) # hacky
    predictions = model.predict(article_batch)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, param_file)
        coord = tf.train.Coordinator()
        thread = threading.Thread(target=load_example, args=(sess, enqueue, coord))
        thread.start()
        tf.train.start_queue_runners(sess=sess)
        
        print >> sys.stderr,  80 * "="
        print >> sys.stderr,  "TESTING"
        print >> sys.stderr,  80 * "="
        with coord.stop_on_exception():
            i = 0
            while True:
                summaries, = sess.run([predictions])
                for summary in summaries.tolist():
                    for id in summary:
                        if id == config.end_token:
                            break
                        print id_to_token[id],
                    print ""
                    i += 1
                    
                    if i >= test_articles.shape[0]:
                        coord.request_stop()
                        coord.join([thread])
                        return



if __name__ == '__main__':
    assert(1 < len(sys.argv) <= 4)
    debug = False
    if 'train' in sys.argv:
        train_main(debug=('debug' in sys.argv), reload_data=('rewrite' in sys.argv))
    elif 'test' in sys.argv:
        test_main(sys.argv[2], debug=('debug' in sys.argv))
    else:
        print >> sys.stderr, "please specify your model: \"train\" or \"test\""
