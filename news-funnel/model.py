import tensorflow as tf
import numpy as np

from model import Model



class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_features = 36
    vocab_size = None
    context_size = 5 # taken from Rush (C)
    summary_length = 9
    article_length = 15
    embed_size = 200 # taken from Rush (D)
    hidden_size = 400 # taken from Rush (H)
    batch_size = 64 # taken from Rush
    n_epochs = 10
    n_layers = 3 # taken from Rush (L)
    lr = 0.05 # taken from Rush
    smoothing_window = 2 # taken from Rush (Q)
    beam_size = 5
    start_token = 0 # Use a rare word as the start token



class RushModel(Model):

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


    def add_embedding(self):
        """
        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        #self.embeddings = tf.get_variable("E", self.word2vec_embeddings)
        # batch_embeddings = tf.nn.embedding_lookup(params=embeddings, ids=self.input_placeholder)
        # embeddings = tf.reshape(batch_embeddings, [-1, self.config.n_features * self.config.embed_size])
        return embeddings


    def add_prediction_op(self):
        """
        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """

        logits = []
        summary = []
        for i in range(self.config.summary_length):
        	context = ([start_token]*self.config.context_size + summary)[-self.config.context_size:]
            
            logits.append(pred)
        	summary.append(tf.argmax(pred, axis=1))
        
        self.logits = tf.stack(logits, axis=1)
        self.summary = summary
    
    def do_prediction_step(input, context):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()

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
            pass

    def add_loss_op(self):
        logits = []
        padded_context = tf.stack(tf.tile(self.config.start_token, [self.config.context_size]), self.summaries_placeholder)
        for i in range(self.config.summary_length):
            context = tf.slice(padded_context, i, self.config.context_size)
            logits.append(do_prediction_step(self.input_placeholder, context))
        logits = tf.stack(logits, axis=1)
    
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.summaries_placeholder))
        
    def predict(self):
        padded_predictions = tf.tile(self.config.start_token, [self.config.batch_size, self.config.context_size])
        for i in range(self.config.summary_length):
            contexts = tf.slice(padded_predictions, [0, i], [-1, self.config.context_size])
            logits = do_prediction_step(self.input_placeholder, contexts)
            padded_predictions = tf.stack([padded_predictions, tf.nn.arg_max(logits)], axis=-1)
        return tf.slice(padded_predictions, [0, self.config.context_size], [-1, -1])
        
        """
        # Beam search, to be completed
        padded_predictions = tf.tile(self.config.start_token, [self.config.beam_size, self.config.batch_size, self.config.context_size])
        prediction_logits = tf.tile(0, [self.config.beam_size, self.config.batch_size, 1])
        for i in range(self.config.summary_length):
            contexts = tf.slice(padded_predictions, [0, 0, i], [-1, -1, self.config.context_size])
            logits = tf.nn.log_softmax(logits=do_prediction_step(self.input_placeholder, contexts))
        
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
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op


	def __init__(self, word2vec_embeddings):
		self.word2vec_embeddings = word2vec_embeddings
        self.config = config
        self.build()



def main(debug=False):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    # prep = preprocess # TODO: take in salient parts of the data (i.e. max title length, article_length, vocab_size)
    embeddings, train_examples, dev_set, test_set = load_and_preprocess_data
    config = Config()



if __name__ == '__main__':
    main()