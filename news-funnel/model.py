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
    context_size = None
    summary_length = 9
    article_length = 15
    dropout = 0.5
    embed_size = 300
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.001



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
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()

        start_token = 0

        output_embeddings = tf.get_variable("E", self.word2vec_embeddings)
        input_embeddings = tf.get_variable("F", self.word2vec_embeddings)
        embedded_input = tf.nn.embedding_lookup(ids=self.input_placeholder, params=input_embeddings)
        
        W = tf.get_variable("W", shape=(self.config.hidden_size, self.config.vocab_size), initializer=xavier_init)
        b1 = tf.get_variable("b1", shape=(self.config.vocab_size,), initializer=zero_init)
        U = tf.get_variable("U", shape=(self.config.hidden_size, self.config.context_size*self.config.embed_size), initializer=xavier_init)
        b2 = tf.get_variable("b2", shape=(self.config.context_size*self.config.embed_size), initializer=zero_init)
        V = tf.get_variable("V",  shape=(self.config.hidden_size, self.config.vocab_size), initializer=xavier_init)

        pred = []
        for i in range(self.config.summary_length):
        	context = ([start_token]*self.config.context_size + pred)[:-self.config.context_size]
        	embedded_context = tf.nn.embedding_lookup(ids=context, params=self.output_embeddings)
        	encoded = self.encode(embedded_input, embedded_context)

        	h = tf.tanh(tf.matmul(y_c, U) + b2)
        	logits = tf.matmul(h, V) + tf.matmul(encoded, W) + b1
        	pred.append(tf.argmax(logits, axis=1))

        return pred


    def encode(self, embedded_input, embedded_context, method="BOW"):
        if method == "BOW":
            return tf.sum(embedded_input, axis=1) / self.config.article_length

	def __init__(self, word2vec_embeddings):
		self.word2vec_embeddings = word2vec_embeddings
        self.config = config
        self.build()





def main(debug=False):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    # prep = preprocess # TODO: take in salient parts of the data (i.e. max title length, article_length, vocab_size)
    config = Config()



if __name__ == '__main__':
    main()