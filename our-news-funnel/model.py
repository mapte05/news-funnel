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
import random
import sys
import os
import glob
import shutil
from utils.model_utils import load_embeddings, load_data, preprocess_data, count_words


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
    hidden_size = 200 # taken from Rush (H = 400)
    batch_size = 512 # Rush uses 64
    #n_epochs = 15 # taken from Rush
    #n_layers = 3 # taken from Rush (L)
    lr = 0.005 # taken from Rush
    lr_decay_base = .5 # taken from Rush
    lr_decay_after_steps = 100000
    lr_staircase = False
    smoothing_window = 2 # taken from Rush (Q)
    beam_size = 5
    encoding_method = "attention" # "attention" or "bag-of-words"
    
    test_interval = 2500
    renormalize_interval = 7800
    
    max_vocab = 75000 # Nallapati 150k
    max_train_articles = None
    max_grad_norm = 5
    
    # Limits for memory conserve
    max_summary_length = 12 #26
    max_article_length = 24 #96
    
    start_token = None # set during preprocessing
    end_token = None # set during preprocessing
    null_token = None # set during preprocessing
    unknown_token = None # set during preprocessing

    num_batches_for_testing = 3


    saver_path = 'variables/news-funnel-model'
    variables_dir = './variables/'
    train_article_file = '../news-funnel/data/train/train.article.txt'
    train_title_file = '../news-funnel/data/train/train.title.txt'
    preprocessed_articles_file="preprocessed_articles_file.npy"
    preprocessed_summaries_file="preprocessed_summaries_file.npy"
    preprocessed_word_distribution_file="preprocessed_word_distribution_file.npy"

    train_loss_file = "eval/train_losses"
    test_loss_file = "eval/test_losses"
    test_results_file_root = "eval/giga_system"
    
    #train_article_file = '../data/train/valid.article.filter.txt' # for debug
    #train_title_file = '../data/train/valid.title.filter.txt' # for debug
    #preprocessed_articles_file="preprocessed_articles_file_valid.npy"
    #preprocessed_summaries_file="preprocessed_summaries_file_valid.npy"
    
    dev_article_file = '../news-funnel/data/train/valid.article.filter.txt'
    dev_title_file = '../news-funnel/data/train/valid.title.filter.txt'
    test_article_file = '../news-funnel/data/giga/input.txt' # also need to test on duc2003/duc2004
    embedding_file = '../news-funnel/data/glove.6B.200d.txt' #TODO: replace with 'glove.6B.200d.txt

class RushModel:

    def __init__(self, config, word2vec_embeddings=None, word_distribution=None):
        if word2vec_embeddings is not None:
            self.word2vec_embeddings = tf.nn.l2_normalize(word2vec_embeddings, 1)
        else:
            self.word2vec_embeddings = None
            
        if word_distribution is not None:
            self.word_distribution = np.log((word_distribution + 1).astype(np.float32))
        else:
            self.word_distribution = None
            
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
    
    def renormalize(self):
        with tf.variable_scope("prediction_step", reuse=True):
            embeddings = [tf.get_variable(name) for name in ["E", "F", "G"]]
            return [E.assign(tf.nn.l2_normalize(E, 1)) for E in embeddings]
    
    def do_prediction_step(self, input, context, suppress_unknown=False):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.constant_initializer(0.0)
        embed_init = self.word2vec_embeddings
        
        # Our custom initializations
        logits_bias_init = self.word_distribution
        attention_bias_init = -0.22 * np.array(range(self.config.article_length), dtype=np.float32) # Decaying attention
        decoder_init = tf.concat_v2([
            embed_init, 
            embed_init, 
            xavier_init((self.config.vocab_size, self.config.hidden_size))
        ], 1) if embed_init is not None else None
        attention_init = tf.concat_v2([np.eye(self.config.embed_size, dtype=np.float32)] * self.config.context_size, 1)

        with tf.variable_scope("prediction_step", reuse=self.defined):
            output_embeddings = tf.get_variable("E", initializer=embed_init)
            input_embeddings = tf.get_variable("F", initializer=embed_init)
            encoding_embeddings = tf.get_variable("G", initializer=embed_init)
            
            embedded_input = tf.nn.embedding_lookup(ids=input, params=input_embeddings)
            embedded_context = tf.reshape(tf.nn.embedding_lookup(ids=context, params=output_embeddings), (-1, self.config.context_size*self.config.embed_size))
            embedded_context_for_encoding = tf.reshape(tf.nn.embedding_lookup(ids=context, params=encoding_embeddings), (-1, self.config.context_size*self.config.embed_size))
            
            U = tf.get_variable("U", shape=(self.config.context_size*self.config.embed_size, self.config.embed_size+self.config.hidden_size), initializer=xavier_init, dtype=tf.float32)
            b1 = tf.get_variable("b1", shape=(1, self.config.embed_size+self.config.hidden_size), initializer=zero_init, dtype=tf.float32)
            
            #V = tf.get_variable("V",  shape=(self.config.vocab_size, self.config.hidden_size + self.config.embed_size), initializer=xavier_init, dtype=tf.float32)
            V = tf.get_variable("V",  initializer=decoder_init)
            #b2 = tf.get_variable("b2", shape=(1, self.config.vocab_size), initializer=logits_bias_init)
            b2 = tf.get_variable("b2", initializer=logits_bias_init)

            #P = tf.get_variable("P", shape=(self.config.embed_size, self.config.embed_size*self.config.context_size), initializer=xavier_init, dtype=tf.float32)
            #b3 = tf.get_variable("b3", shape=(1, self.config.article_length), initializer=attention_bias_init)
            P = tf.get_variable("P", initializer=attention_init)
            b3 = tf.get_variable("b3", initializer=attention_bias_init)
            self.defined = True
        
            if self.config.encoding_method == "bag-of-words":
                encoded = tf.reduce_mean(embedded_input, axis=-2) # average along input
            elif self.config.encoding_method == "attention":
                p = tf.nn.softmax(tf.einsum('ij,bwi,bj->bw', P, embedded_input, embedded_context_for_encoding) + b3)
                
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
            x = tf.concat_v2([h, encoded], 1)
            logits = tf.matmul(x, tf.transpose(V)) + b2
            return logits, x

    def add_loss_op(self, articles, summaries):
        logits = []
        padded_context = tf.concat_v2([
            tf.fill([self.config.batch_size, self.config.context_size], self.config.start_token), 
            summaries], 1)
        for i in xrange(self.config.summary_length):
            context = tf.slice(padded_context, [0, i], [-1, self.config.context_size])
            logit, _ = self.do_prediction_step(articles, context)
            logits.append(logit)
        logits = tf.stack(logits, axis=1)
        
        null_mask = tf.not_equal(summaries, self.config.null_token)
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=summaries)
        return tf.reduce_mean(tf.boolean_mask(cross_entropy_loss, null_mask))
        
    def add_approx_loss_op(self, articles, summaries):
        activations = []
        padded_context = tf.concat_v2([
            tf.fill([self.config.batch_size, self.config.context_size], self.config.start_token), 
            summaries], 1)
        for i in xrange(self.config.summary_length):
            context = tf.slice(padded_context, [0, i], [-1, self.config.context_size])
            _, x = self.do_prediction_step(articles, context)
            activations.append(x)
        
        with tf.variable_scope("prediction_step", reuse=True):
            V = tf.get_variable("V")
            b = tf.get_variable("b2")
        
        activations = tf.reshape(tf.stack(activations, axis=1), (self.config.batch_size*self.config.summary_length, -1))
        summaries = tf.reshape(summaries, (self.config.batch_size*self.config.summary_length, -1))
        null_mask = tf.squeeze(tf.not_equal(summaries, self.config.null_token))
        
        # NOTE: r0.12 -> r1.0 swaps arg order
        cross_entropy_loss = tf.nn.sampled_softmax_loss(V, b, activations, summaries, 2048, self.config.vocab_size)
        return tf.reduce_mean(tf.boolean_mask(cross_entropy_loss, null_mask))
        
    def predict(self, articles, method="beam"):
        if method == "greedy":
            padded_predictions = tf.fill([self.config.batch_size, self.config.context_size], self.config.start_token)
            for i in range(self.config.summary_length):
                context = tf.slice(padded_predictions, [0, i], [-1, self.config.context_size])
                logits, _ = self.do_prediction_step(articles, context, suppress_unknown=True)
                
                # Experiment: use only common words
                #logits = tf.slice(logits, [0, 0], [-1, 30000])
                
                padded_predictions = tf.concat_v2([padded_predictions, tf.expand_dims(tf.to_int32(tf.argmax(logits, axis=1)), -1)], 1)
            return tf.slice(padded_predictions, [0, self.config.context_size], [-1, -1])
        
        elif method == "beam":            
            padded_predictions = tf.fill([self.config.batch_size, self.config.beam_size, self.config.context_size], self.config.start_token)
            prediction_log_probs = tf.fill([self.config.batch_size, self.config.beam_size], 0)
            for i in range(self.config.summary_length):
                contexts = tf.slice(padded_predictions, [0, 0, i], [-1, -1, self.config.context_size])
                
                contexts = tf.transpose(contexts, [1,0,2])
                logits = tf.map_fn((lambda context: self.do_prediction_step(articles, context)[0]), contexts)
                logits = tf.transpose(logits, [1,0,2])
                assert logits.get_shape() == (self.config.batch_size, self.config.beam_size, self.config.vocab_size)
                
                log_probs = prediction_log_probs + tf.nn.log_softmax(logits=logits)
                assert log_probs.get_shape() == (self.config.batch_size, self.config.beam_size, self.config.vocab_size)
            
                collapsed_log_probs = tf.reshape(log_probs, (self.config.batch_size, self.config.beam_size*self.config.vocab_size))
                prediction_log_probs, indices = tf.nn.top_k(input=collapsed_log_probs, k=self.config.beam_size) 
                best_words = tf.mod(indices, self.config.vocab_size)
                best_beams = tf.div(indices, self.config.vocab_size)
                assert prediction_log_probs.get_shape() == (self.config.batch_size, self.config.beam_size)
                assert best_words.get_shape() == (self.config.batch_size, self.config.beam_size)
                assert best_beams.get_shape() == (self.config.batch_size, self.config.beam_size)
                
                padded_predictions = tf.concat_v2([
                    tf.gather_nd(padded_predictions, best_beams),
                    best_words
                ], 2)
            
            return tf.squeeze(tf.slice(padded_predictions, [0, 0, self.config.context_size], [-1, 1, -1]), [1])
    
        else:
            raise Exception("predict method not greedy or beam")

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        global_step = tf.get_variable("global_step", initializer=0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.config.lr, global_step, self.config.lr_decay_after_steps, self.config.lr_decay_base, staircase=self.config.lr_staircase)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        #train_op = optimizer.minimize(loss, global_step=global_step)
        grads, vars = zip(*optimizer.compute_gradients(loss))

        grads_clipped, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
        train_op = optimizer.apply_gradients(zip(grads_clipped, vars), global_step=global_step)

        return train_op, global_step, tf.global_norm(grads), learning_rate

def write_config(config, config_file):
    with open(config_file, 'wb') as outf:
        pickle.dump(config, outf, pickle.HIGHEST_PROTOCOL)

def load_config(config_file):
    with open(config_file, 'rb') as inpf:
        config = pickle.load(inpf)
    return config


def train_main(config_file="config/config_file", debug=True, reload_data=False, load_vars_from_file=False): 
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
        word_distribution = np.load(config.preprocessed_word_distribution_file)

        config.article_length = train_articles.shape[1]
        config.summary_length = train_summaries.shape[1]
    else:
        train_articles = load_data(config.train_article_file, config.max_train_articles)
        config.article_length = min(max(len(x) for x in train_articles) + 1, config.max_article_length)
        train_articles = preprocess_data(train_articles, token_to_id, config.article_length)
        
        train_summaries = load_data(config.train_title_file, config.max_train_articles)
        config.summary_length = min(max(len(x) for x in train_summaries) + 1, config.max_summary_length)
        train_summaries = preprocess_data(train_summaries, token_to_id, config.summary_length)
        
        word_distribution = count_words(train_summaries, config.vocab_size, config.null_token)

        if not debug:
        	np.save(config.preprocessed_articles_file, train_articles)
	        np.save(config.preprocessed_summaries_file, train_summaries)
	        np.save(config.preprocessed_word_distribution_file, word_distribution)
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
            shuffled = range(train_articles.shape[0])
            random.shuffle(shuffled)
            for i in shuffled:
                sess.run(enqueue, feed_dict={train_article_input: train_articles[i], train_summary_input: train_summaries[i]})
                if coord.should_stop():
                    return

    def load_dev_example(sess, enqueue, coord):
        while True:
            for i in xrange(config.num_batches_for_testing * config.batch_size):
                sess.run(enqueue, feed_dict={dev_article_input: dev_articles[i], dev_summary_input: dev_summaries[i]})
                if coord.should_stop():
                    return

    model = RushModel(config, embeddings, word_distribution)
    
    # Define training pipeline
    train_article_input = tf.placeholder(tf.int32, shape=(config.article_length,))
    train_summary_input = tf.placeholder(tf.int32, shape=(config.summary_length,))
    train_queue = tf.FIFOQueue(1024, [tf.int32, tf.int32], shapes=[(config.article_length,), (config.summary_length,)])
    train_enqueue = train_queue.enqueue([train_article_input, train_summary_input])
    
    train_article_batch, train_summary_batch = train_queue.dequeue_many(config.batch_size)
    train_article_batch = tf.reshape(train_article_batch, (config.batch_size, config.article_length)) # hacky
    train_summary_batch = tf.reshape(train_summary_batch, (config.batch_size, config.summary_length))
    
    train_loss_op = model.add_approx_loss_op(train_article_batch, train_summary_batch)
    training_op, global_step, grad_norm_op, lr_op = model.add_training_op(train_loss_op)
    renormalize_op = model.renormalize()

    # Define testing pipeline
    dev_article_input = tf.placeholder(tf.int32, shape=(config.article_length,))
    dev_summary_input = tf.placeholder(tf.int32, shape=(config.summary_length,))
    dev_queue = tf.FIFOQueue(1024, [tf.int32, tf.int32], shapes=[(config.article_length,), (config.summary_length,)])
    dev_enqueue = dev_queue.enqueue([dev_article_input, dev_summary_input])
    
    dev_article_batch, dev_summary_batch = dev_queue.dequeue_many(config.batch_size)
    dev_article_batch = tf.reshape(dev_article_batch, (config.batch_size, config.article_length)) # hacky
    dev_summary_batch = tf.reshape(dev_summary_batch, (config.batch_size, config.summary_length))
    
    dev_approx_loss_op = model.add_approx_loss_op(dev_article_batch, dev_summary_batch)
    dev_loss_op = model.add_loss_op(dev_article_batch, dev_summary_batch)
    predictions = model.predict(dev_article_batch)

    if load_vars_from_file:
        tlossf = open(config.test_loss_file, "a+")
        lf = open(config.train_loss_file, "a+")
    else:
        tlossf = open(config.test_loss_file, "w+")
        lf = open(config.train_loss_file, "w+")

    def test_lite(sess, count):
        with open(config.test_results_file_root+str(count), 'w+') as testf:
            loss_sum = 0.
            approx_loss_sum = 0.
            for i in xrange(config.num_batches_for_testing):
                summaries, loss, approx_loss = sess.run([predictions, dev_loss_op, dev_approx_loss_op])
                loss_sum += loss
                approx_loss_sum += approx_loss
                for summary in summaries.tolist():
                    line = []
                    for id in summary:
                        if id == config.end_token:
                            break
                        line.append(id_to_token[id])
                    testf.write(' '.join(word for word in line)+'\n')
            mean_loss = loss_sum / config.num_batches_for_testing
            mean_approx_loss = approx_loss_sum / config.num_batches_for_testing
            grad_norm, lr = sess.run([grad_norm_op, lr_op])
            tlossf.write(','.join([str(count), str(mean_loss), str(grad_norm), str(lr), str(mean_approx_loss)]) + '\n')
            tlossf.flush()
        return loss_sum

    saver = tf.train.Saver(max_to_keep=2)
    with tf.Session() as sess:
        if load_vars_from_file:
            saver.restore(sess, tf.train.latest_checkpoint(config.variables_dir))
        else:
            sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
        
        coord = tf.train.Coordinator()
        threads = [
            threading.Thread(target=load_train_example, args=(sess, train_enqueue, coord)),
            threading.Thread(target=load_dev_example, args=(sess, dev_enqueue, coord))
        ]
        for thread in threads:
            thread.start()

        print 80 * "="
        print "TRAINING"
        print 80 * "="
        best_loss = float('inf')
        # with coord.stop_on_exception():
        start = time.time()
        while True:
            loss, counter, _ = sess.run([train_loss_op, global_step, training_op])
            lf.write(str(counter) + ','+str(loss)+'\n')
            
            if counter % 10 == 0:
                lf.flush()
                print counter, " minibatches took {:.2f} seconds".format(time.time() - start)

            if counter % config.test_interval == 0:
                test_loss = test_lite(sess, counter)
                print "SAVED AND TESTED ON PARAMETERS | loss:", loss, "| counter:", counter
                    
                # Save best model
                if test_loss < best_loss:
                    best_loss = test_loss
                    saver.save(sess, config.saver_path, global_step=counter)
            
            if counter % config.renormalize_interval == 0:
                sess.run(renormalize_op)
                print "RENORMALIZED"



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
    
    model = RushModel(config)

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
        train_main(debug=('debug' in sys.argv), reload_data=('rewrite' in sys.argv), load_vars_from_file=('resume' in sys.argv))
    elif 'test' in sys.argv:
        test_main(sys.argv[2], debug=('debug' in sys.argv))
    else:
        print >> sys.stderr, "please specify your model: \"train\" or \"test\""
