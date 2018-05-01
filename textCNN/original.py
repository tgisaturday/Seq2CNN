import numpy as np
import tensorflow as tf
initializer = tf.contrib.layers.xavier_initializer()
he_normal = tf.keras.initializers.he_normal()
rand_uniform = tf.random_uniform_initializer(-1,1,seed=2)
regularizer = tf.contrib.layers.l2_regularizer(1e-3)
class TextCNN(object):
    def __init__(self,sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 fc_layer_norm=False,temp_norm=True):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.is_training = tf.placeholder(tf.bool, name='is_training')


        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embeddings = tf.get_variable(name='embedding_W', shape=[vocab_size, embedding_size],initializer=rand_uniform)
            self.embedded_chars = tf.nn.embedding_lookup(embeddings, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        self.pooled_outputs=[]
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.get_variable(name='W', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                #Apply nonlinearity
                h = tf.nn.relu(conv, name='relu')
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                       padding='VALID', name='pool')                
                self.pooled_outputs.append(pooled)
 
        self.h_pool = tf.concat(self.pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, len(filter_sizes)*num_filters])
        with tf.variable_scope('output'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            W = tf.get_variable('W', shape=[len(filter_sizes)*num_filters, num_classes],
                                    initializer=initializer,regularizer = regularizer)
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.1))
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')   
        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):            
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            cnn_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,logits=self.scores)
            self.loss = tf.reduce_mean(cnn_loss)+tf.reduce_sum(regularization_losses)

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')