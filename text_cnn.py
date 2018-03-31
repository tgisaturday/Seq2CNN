import numpy as np
import tensorflow as tf


class VGG_text(object):  
    def __init__(self, num_classes, max_length, vocab_size, embedding_size):

        
        self.input_x = tf.placeholder(tf.int32, [None, max_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x) 
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) 
            
        #VGGnet_Bigram
        with tf.name_scope('VGGnet_Bigram'):
            filter_size = 2
            num_filters = max_length-filter_size+1
            
            filter_shape = [2, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv1_1 = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding='SAME', name='conv1_1')
            h1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, b), name='relu1_1')
            
            pool1= tf.nn.max_pool(h1_1, ksize=[1, num_filters, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool1')
            
            filter_shape = [3, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv2_1 = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding='SAME', name='conv2_1')
            h2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, b), name='relu2_1')

            filter_shape = [2, embedding_size, num_filters, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv2_2 = tf.nn.conv2d(h2_1, W, strides=[1, 1, 1, 1], padding='SAME', name='conv2_2')
            h2_2 = tf.nn.relu(tf.nn.bias_add(conv2_2, b), name='relu2_2')
            
            pool2= tf.nn.max_pool(h2_2, ksize=[1,num_filters, 1, 1], strides=[1, 1, 1, 1],padding='SAME', name='pool1')
            
            filter_shape = [4, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv3_1 = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding='SAME', name='conv3_1')
            h3_1 = tf.nn.relu(tf.nn.bias_add(conv3_1, b), name='relu3_1')
            
            filter_shape = [3, embedding_size, num_filters, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv3_2 = tf.nn.conv2d(h3_1, W, strides=[1, 1, 1, 1], padding='SAME', name='conv3_2')
            h3_2 = tf.nn.relu(tf.nn.bias_add(conv3_2, b), name='relu3_2')
            
            filter_shape = [2, embedding_size, num_filters, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv3_3 = tf.nn.conv2d(h3_2, W, strides=[1, 1, 1, 1], padding='SAME', name='conv3_3')
            h3_3 = tf.nn.relu(tf.nn.bias_add(conv3_3, b), name='relu3_3')
            
            pool3 = tf.nn.max_pool(h3_3 , ksize=[1,num_filters, 1, 1], strides=[1, 1, 1, 1],padding='SAME', name='pool3')


            num_filters_total = num_filters* (max_length *embedding_size) *3
            
            total_pools = [pool1,pool2,pool3]    
            self.pool_h = tf.concat(total_pools, 3)
            self.pool_h_flat = tf.reshape(self.pool_h, [-1, num_filters_total])
            
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.pool_h_flat, self.dropout_keep_prob)
                
        with tf.name_scope('output'):
            W = tf.get_variable('W', shape=[num_filters_total, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')
                
        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,logits=self.scores) 
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
            
