'''
correct use

training: python model.py train
testing: python model.py test param_file

'''
import tensorflow as tf
import numpy as np
import time
import pickle
import sys
from utils.model_utils import load_embeddings, load_data, preprocess_data


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_features = 36
    vocab_size = None # set during preprocessing
    context_size = 5 # taken from Rush (C)
    summary_length = None # set during preprocessing
    article_length = None # set during preprocessing
    embed_size = 200 # taken from Rush (D)
    hidden_size = 400 # taken from Rush (H)
    batch_size = 64 # taken from Rush
    n_epochs = 15 # taken from Rush
    n_layers = 3 # taken from Rush (L)
    lr = 0.05 # taken from Rush
    smoothing_window = 2 # taken from Rush (Q)
    beam_size = 5
    start_token = None # set during preprocessing

    saver_path = 'variables/news-funnel-model'
    train_article_file = './data/train/valid.article.filter.txt' # TODO: replace at actual training time with 'train/train.article.txt'
    train_title_file = './data/train/valid.title.filter.txt' # TODO: replace at actual training time with 'train/train.title.txt'
    dev_article_file = './data/train/valid.article.filter.txt'
    dev_title_file = './data/train/valid.title.filter.txt'
    test_article_file = './data/giga/input.txt' # also need to test on duc2003/duc2004
    test_title_file = './data/giga/task1_ref0.txt'
    embedding_file = './data/glove.6B.50d.txt' #TODO: replace with 'glove.6B.200d.txt
    

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
    
    def do_prediction_step(self, input, context):
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
            self.defined = True

            h = tf.tanh(tf.matmul(embedded_context, U) + b1)
            encoded = self.encode(embedded_input, embedded_context_for_encoding)
            logits = tf.matmul(h, V) + tf.matmul(encoded, W) + b2
        
            return logits

    def encode(self, embedded_input, embedded_context, method="BOW"):
        if method == "BOW":
            return tf.reduce_mean(embedded_input, axis=-2) # average along input
        if method == "ATT":
            raise NotImplementedError

    def add_loss_op(self, articles, summaries):
        logits = []
        padded_context = tf.concat_v2([
            tf.fill([self.config.batch_size, self.config.context_size], self.config.start_token), 
            summaries], 1)
        for i in range(self.config.summary_length):
            context = tf.slice(padded_context, [0, i], [-1, self.config.context_size])
            logits.append(self.do_prediction_step(articles, context))
        logits = tf.stack(logits, axis=1)
    
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=summaries))
        
    def predict(self, articles, method="beam"):
        if method == "greedy":
            padded_predictions = tf.fill(self.config.start_token, [self.config.batch_size, self.config.context_size])
            for i in range(self.config.summary_length):
                context = tf.slice(padded_predictions, [0, i], [-1, self.config.context_size])
                logits = self.do_prediction_step(articles, context)
                padded_predictions = tf.concat_v2([padded_predictions, tf.argmax(logits, axis=-1)], 1)
            return tf.slice(padded_predictions, [0, self.config.context_size], [-1, -1])
        """
        elif method == "beam":
            padded_predictions = tf.fill(self.config.start_token, [self.config.batch_size, self.config.beam_size, self.config.context_size])
            prediction_log_probs = tf.fill(0, [self.config.batch_size, self.config.beam_size])
            for i in range(self.config.summary_length):
                context = tf.slice(padded_predictions, [0, 0, i], [-1, -1, self.config.context_size])
                
                log_probs = prediction_log_probs + tf.nn.log_softmax(logits=do_prediction_step(articles, context)) 
                assert log_probs.get_shape() == (self.config.batch_size, self.config.beam_size, self.config.vocab_size)
            
                best_log_probs, best_indices = tf.nn.top_k(input=log_probs, k=self.config.beam_size) 
                
                tf.reshape(best_log_probs, (self.batch_size, self.beam_search**2))
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
        else:
            raise Exception("predict method not greedy or beam")

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        return tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        


def write_config(config, config_file):
    with open(config_file, 'wb') as outf:
        pickle.dump(config, outf, pickle.HIGHEST_PROTOCOL)

def load_config(config_file):
    with open(config_file, 'rb') as inpf:
        config = pickle.load(inpf)
    return config


def train_main(config_file="config/config_file", debug=True, run_dev=False):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()

    print "Loading embedding data...",
    start = time.time()
    embeddings, token_to_id, id_to_token = load_embeddings(config.embedding_file, debug=debug)
    config.vocab_size = embeddings.shape[0]
    config.embed_size = embeddings.shape[1]
    config.start_token = token_to_id('<START>')
    print "loaded {0} embeddings".format(config.vocab_size)
    print "took {:.2f} seconds".format(time.time() - start)

    print "Loading training data...",
    start = time.time()
    train_articles = load_data(config.train_article_file, debug=debug)
    config.article_length = article_length = max([len(x) for x in train_articles])
    train_articles = preprocess_data(train_articles, token_to_id, article_length)
    
    train_summaries = load_data(config.train_title_file, debug=debug)
    config.summary_length = summary_length = max([len(x) for x in train_summaries])
    train_summaries = preprocess_data(train_summaries, token_to_id, summary_length)
    print "loaded {0} articles, {1} summaries".format(train_articles.shape[0], train_summaries.shape[0])
    print "took {:.2f} seconds".format(time.time() - start)

    if run_dev:
        print "Loading dev data...",
        start = time.time()
        dev_articles = load_data(config.dev_article_file)
        dev_articles = preprocess_data(dev_articles, token_to_id, config.article_length)
        
        dev_summaries = load_data(config.dev_title_file)
        dev_summaries = preprocess_data(dev_summaries, token_to_id, config.summary_length)
        print "took {:.2f} seconds".format(time.time() - start)

    print "writing Config to file"
    write_config(config, config_file)

    model = RushModel(embeddings, config)
    article_batch, summary_batch =  tf.train.shuffle_batch([train_articles, train_summaries], 
        batch_size=config.batch_size,
        num_threads=1,
        capacity=50000,
        min_after_dequeue=10,   # TODO make bigger during real stuff
        enqueue_many=True)
    loss_op = model.add_loss_op(article_batch, summary_batch)
    training_op = model.add_training_op(loss_op)
            
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        counter = 0

        print 80 * "="
        print "TRAINING"
        print 80 * "="
        for epoch in range(config.n_epochs):
            tf.train.start_queue_runners(sess=sess)
            try:
                while True:
                    counter += 1
                    if counter % 1000 == 0:
                        saver.save(sess, config.saver_path, global_step=counter)
                    loss, _ = sess.run([loss_op, training_op])
                    print "loss:", loss
            except tf.errors.OutOfRangeError:
                print "end of epoch #", epoch, "..."



def test_main(param_file, config_file="config/config_file", load_config_from_file=True, debug=False):
    config = None
    if load_config_from_file:
        config = load_config(config_file)
    else:
        config = Config()

    print >> sys.stderr, "Loading embedding data...",
    start = time.time()
    embeddings, token_to_id, id_to_token = load_embeddings(config.data_path, config.embedding_file)
    assert len(embeddings) == config.vocab_size
    print >> sys.stderr,  "took {:.2f} seconds".format(time.time() - start)

    print >> sys.stderr, "Loading test data...",
    start = time.time()
    
    test_articles = load_data(config.test_article_file)
    test_articles = preprocess_data(test_articles, token_to_id, config.article_length)
    print >> sys.stderr, "took {:.2f} seconds".format(time.time() - start)
    
    model = RushModel(embeddings, config)
    article_batch = tf.train.batch([test_articles],
        batch_size=config.batch_size,
        num_threads=1, 
        enqueue_many=True,
        allow_smaller_final_batch=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, param_file)
        tf.train.start_queue_runners(sess=sess)
        try:
            while True:
                summaries = sess.run([model.predict(article_batch)])
                for summary in summaries:
                    print ' '.join(id_to_token(id) for id in summary)
        except tf.errors.OutOfRangeError:
            pass



if __name__ == '__main__':
    assert(len(sys.argv) == 2 or len(sys.argv) == 3)
    if sys.argv[1] == "train":
        train_main()
    elif sys.argv[1] == "test":
        test_main(arg[2])
    else:
        print "please specify your model: \"train\" or \"test\""
