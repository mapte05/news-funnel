'''
correct use

training: python model.py train config/config_file
testing: python model.py test config/config_file

'''
import tensorflow as tf
import numpy as np
import time
import pickle
import sys


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

    data_path = './data'
    train_article_file = 'train/valid.article.filter.txt' # TODO: replace at actual training time with 'train/train.article.txt'
    train_title_file = 'train/valid.title.filter.txt' # TODO: replace at actual training time with 'train/train.title.txt'
    dev_article_file = 'train/valid.article.filter.txt'
    dev_title_file = 'train/valid.title.filter.txt'
    test_article_file = 'giga/input.txt' # also need to test on duc2003/duc2004
    test_title_file = 'giga/task1_ref0.txt'
    embedding_file = 'glove.6B.50d.txt' #TODO: replace with 'glove.6B.200d.txt
    

class RushModel:

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
    
    def do_prediction_step(input, context):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()

        with tf.variable_scope("prediction_step", reuse=True):
            output_embeddings = tf.get_variable("E", self.word2vec_embeddings)
            input_embeddings = tf.get_variable("F", self.word2vec_embeddings)
            encoding_embeddings = tf.get_variable("G", self.word2vec_embeddings)
            
            embedded_input = tf.nn.embedding_lookup(ids=input, params=input_embeddings)
            embedded_context = tf.reshape(tf.nn.embedding_lookup(ids=context, params=self.output_embeddings), (-1, self.config.context_size*self.config.embed_size))
            embedded_context_for_encoding = tf.reshape(tf.nn.embedding_lookup(ids=context, params=self.encoding_embeddings), (-1, self.config.context_size*self.config.embed_size))
            
            U = tf.get_variable("U", shape=(self.config.context_size*self.config.embed_size, self.config.hidden_size), initializer=xavier_init)
            b1 = tf.get_variable("b1", shape=(1, self.config.hidden_size), initializer=zero_init)
            
            V = tf.get_variable("V",  shape=(self.config.hidden_size, self.config.vocab_size), initializer=xavier_init)
            W = tf.get_variable("W", shape=(self.config.hidden_size, self.config.vocab_size), initializer=xavier_init)
            b2 = tf.get_variable("b2", shape=(1, self.config.vocab_size), initializer=zero_init)
            
            h = tf.tanh(tf.matmul(embedded_context, U) + b1)
            encoded = self.encode(embedded_input, embedded_context_for_encoding)
            logits = tf.matmul(h, V) + tf.matmul(encoded, W) + b2
        
            return logits

    def encode(self, embedded_input, embedded_context, method="BOW"):
        if method == "BOW":
            return tf.reduce_mean(embedded_input, axis=1)
        if method == "ATT":
            raise NotImplementedError

    def add_loss_op(self, articles, summaries):
        logits = []
        padded_context = tf.stack(tf.tile(self.config.start_token, [self.config.context_size]), summaries)
        for i in range(self.config.summary_length):
            context = tf.slice(padded_context, i, self.config.context_size)
            logits.append(do_prediction_step(articles, context))
        logits = tf.stack(logits, axis=1)
    
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=summaries))
        
    def predict(self, articles):
        padded_predictions = tf.tile(self.config.start_token, [self.config.batch_size, self.config.context_size])
        for i in range(self.config.summary_length):
            contexts = tf.slice(padded_predictions, [0, i], [-1, self.config.context_size])
            logits = do_prediction_step(articles, contexts)
            padded_predictions = tf.stack([padded_predictions, tf.nn.arg_max(logits)], axis=-1)
        return tf.slice(padded_predictions, [0, self.config.context_size], [-1, -1])
        
        """
        # Beam search, to be completed
        padded_predictions = tf.tile(self.config.start_token, [self.config.beam_size, self.config.batch_size, self.config.context_size])
        prediction_logits = tf.tile(0, [self.config.beam_size, self.config.batch_size, 1])
        for i in range(self.config.summary_length):
            contexts = tf.slice(padded_predictions, [0, 0, i], [-1, -1, self.config.context_size])
            logits = tf.nn.log_softmax(logits=do_prediction_step(articles, contexts))
        
            logits_beam, indices_beam = tf.nn.top_k(input=logits, k=self.config.beam_size)
            padded_predictions = tf.stack()
            prediction_logits = 
        """

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        return tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)

    def get_training_batch(self, articles, summaries):
        return tf.train.shuffle_batch([articles, summaries], 
            batch_size=self.config.batch_size,
            num_threads=1,
            capacity=50000,
            min_after_dequeue=10000,
            enqueue_many=True)

    def train(self, articles, summaries, epoch_limit):
        #epochs = tf.Variable(0)
        #tf.count_up_to(epochs, epoch_limit)
    
        article_batch, sumary_batch = get_training_batch(self, articles, summaries)
        loss_op = self.add_loss_op(article_batch, sumary_batch)
        training_op = self.add_training_op(loss_op)
    
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            tf.train.start_queue_runners(sess=sess)
            while True:
                loss, _ = sess.run([loss, training_op])
                print loss
                

	def __init__(self, word2vec_embeddings):
		self.word2vec_embeddings = word2vec_embeddings
        self.config = config
        
        self.add_placeholders()
        


def write_config(config, config_file):
    with open(config_file, 'wb') as outf:
        pickle.dump(config, outf, pickle.HIGHEST_PROTOCOL)

def load_config(config_file):
    with open(config_file, 'rb') as inpf:
        config = pickle.load(inpf)
    return config


def train_main(config_file, debug=False, run_dev=False):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()

    print "Loading embedding data...",
    start = time.time()
    embeddings, token_to_id, id_to_token = load_embeddings(config.data_path, config.embedding_file)
    config.vocab_size = len(embeddings)
    config.start_token = token_to_id('<START>')
    print "took {:.2f} seconds".format(time.time() - start)

    print "Loading training data...",
    start = time.time()
    train_articles, train_summaries = load_data(config.data_path, config.train_article_file, config.train_title_file)
    config.article_length = article_length = max([len(x) for x in train_articles])
    config.summary_length = summary_length = max([len(x) for x in train_summaries])
    train_articles, train_summaries = preprocess_data(train_articles, train_summaries, token_to_id, article_length, summary_length)
    print "took {:.2f} seconds".format(time.time() - start)

    if run_dev:
        print "Loading dev data...",
        start = time.time()
        dev_articles, dev_summaries = load_data(config.data_path, config.dev_article_file, config.dev_title_file)
        dev_articles, dev_summaries = preprocess_data(dev_articles, dev_summaries, token_to_id, config.article_length, config.summary_length)
        print "took {:.2f} seconds".format(time.time() - start)

    print "writing Config to file"
    write_config(config, config_file)



def test_main(config_file, param_file, load_config_from_file=True, debug=False):

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
    # TODO: break up load_data and preprocess_data
    test_articles, test_summaries = load_data(config.data_path, config.test_article_file, config.test_title_file)
    test_articles, test_summaries = preprocess_data(test_articles, test_summaries, token_to_id, config.article_length, config.summary_length)
    print >> sys.stderr, "took {:.2f} seconds".format(time.time() - start)
    
    model = RushModel(param_file=param_file)
    article_batch = tf.train.batch([test_articles],
        batch_size=config.batch_size,
        num_threads=1, 
        enqueue_many=True,
        allow_smaller_final_batch=True)
    
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)
        try:
            while True:
                summaries = sess.run([model.predict(article_batch)])
                for summary in summaries:
                    print ' '.join(id_to_token(id) for id in summary)
        except tf.errors.OutOfRangeError:
            pass

if __name__ == '__main__':
    assert(len(args) == 3) # third argument is path to config file
    if args[1] == "train":
        train_main(args[2])
    elif args[1] == "test":
        test_main(args[2], arg[3])
    else:
        print "please specify your model: \"train\" or \"test\""
