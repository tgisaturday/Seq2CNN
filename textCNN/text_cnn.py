import numpy as np
import tensorflow as tf


class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0,fc_layer_norm=True,temp_norm=False):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(self.embedded_chars_expanded , W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
                #Apply nonlinearity
                if temp_norm:
                    h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
                    h = tf.nn.relu(tf.nn.bias_add(h, b), name='relu')
                else:
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length, 1, 1], strides=[1, 1, 1, 1],
                                        padding='SAME', name='pool')
                    
                filter_shape = [3, 1, num_filters, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
                if temp_norm:
                    h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
                    h = tf.nn.relu(tf.nn.bias_add(h, b), name='relu')
                else:
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # Maxpooling over the outputs                        
                pooled = tf.nn.max_pool(h, ksize=[1, 1 , 1, 1], strides=[1, 1, 1, 1],
                                        padding='SAME', name='pool') 
                    
                pooled_outputs.append(pooled)
            # Combine all the pooled features
        with tf.name_scope('fc-dropout-5'):
            num_filters_total = num_filters * 3 
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, sequence_length*num_filters_total*embedding_size])
            W = tf.Variable(tf.truncated_normal([sequence_length*num_filters_total*embedding_size, num_filters_total ], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name='b')
            if fc_layer_norm:
                h_drop = tf.contrib.layers.batch_norm(h_pool_flat,center=True, scale=True,is_training=self.is_training)
            else:
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
                
            h =tf.nn.relu( tf.nn.xw_plus_b(h_drop, W, b, name='relu'))
            
        with tf.name_scope('fc-dropout-6'):
            h_pool_flat = tf.reshape(h, [-1, num_filters_total])
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name='b')
            if fc_layer_norm:
                h_drop = tf.contrib.layers.batch_norm(h_pool_flat,center=True, scale=True,is_training=self.is_training)
            else:
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
            h =tf.nn.relu( tf.nn.xw_plus_b(h_drop, W, b, name='relu'))
            
        with tf.name_scope('fc-dropout-7'):            
            h_pool_flat = tf.reshape(h, [-1, num_filters_total])
            if fc_layer_norm:
                h_drop = tf.contrib.layers.batch_norm(h_pool_flat,center=True, scale=True,is_training=self.is_training)
            else:
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)      
            W = tf.get_variable('W', shape=[num_filters_total, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer(),regularizer = regularizer)
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            self.scores = tf.nn.xw_plus_b(h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')
        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,
                                                             logits=self.scores)  # only named arguments accepted
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
