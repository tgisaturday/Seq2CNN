import numpy as np
import tensorflow as tf
initializer = tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer(1e-3)

class TextCNN(object):
    def __init__(self,sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0,fc_layer_norm=False,temp_norm=True):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size

        self.pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.get_variable(name='W', shape=filter_shape,initializer=initializer,regularizer=regularizer)
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
                #Apply nonlinearity
                if temp_norm:
                    h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
                    h = tf.nn.relu(h, name='relu')
                else:
                    h = tf.nn.relu(conv, name='relu')
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length, 1, 1], strides=[1, 1, 1, 1],
                                        padding='SAME', name='pool')
                    
                filter_shape = [3, 1, num_filters, num_filters]
                W_2 = tf.get_variable(name='W_2', shape=filter_shape,initializer=initializer,regularizer=regularizer)
                conv = tf.nn.conv2d(pooled, W_2, strides=[1, 1, 1, 1], padding='SAME', name='conv')
                if temp_norm:
                    h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
                    h = tf.nn.relu(h, name='relu')
                else:
                    h = tf.nn.relu(conv, name='relu')
                # Maxpooling over the outputs                        
                pooled = tf.nn.max_pool(h, ksize=[1, 1 , 1, 1], strides=[1, 1, 1, 1],
                                        padding='SAME', name='pool') 
                    
                self.pooled_outputs.append(pooled)
        # Combine all the pooled features
        with tf.variable_scope('fc-dropout-5'):
            h_pool = tf.concat(self.pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, len(filter_sizes)*embedding_size*sequence_length*num_filters])
            W = tf.get_variable('W', shape=[len(filter_sizes)*sequence_length*embedding_size*num_filters, len(filter_sizes)*num_filters],
                                    initializer=initializer,regularizer = regularizer)
            b = tf.get_variable('b', [len(filter_sizes)*num_filters], initializer=tf.constant_initializer(1.0))
            fc =  tf.nn.xw_plus_b(h_pool_flat, W, b, name='fc5')
            if fc_layer_norm:
                fc = tf.contrib.layers.batch_norm(fc,center=True, scale=True,is_training=self.is_training)                
            relu =tf.nn.relu(fc)
            self.fc5 = tf.nn.dropout(relu, self.dropout_keep_prob)
            
        with tf.variable_scope('fc-dropout-6'):
            W = tf.get_variable('W', shape=[len(filter_sizes)*num_filters, len(filter_sizes)*num_filters],
                                    initializer=initializer,regularizer = regularizer)
            b = tf.get_variable('b', [len(filter_sizes)*num_filters], initializer=tf.constant_initializer(1.0))
            fc =  tf.nn.xw_plus_b(self.fc5, W, b, name='fc6')
            if fc_layer_norm:
                fc = tf.contrib.layers.batch_norm(fc,center=True, scale=True,is_training=self.is_training)                
            relu =tf.nn.relu(fc)
            self.fc6 =tf.nn.dropout(relu, self.dropout_keep_prob)
            
        with tf.variable_scope('fc-dropout-7'):            
            W = tf.get_variable('W', shape=[len(filter_sizes)*num_filters, num_classes],
                                    initializer=initializer,regularizer = regularizer)
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(1.0))
            self.scores = tf.nn.xw_plus_b(self.fc6, W, b, name='fc7')
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