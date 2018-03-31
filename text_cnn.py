import numpy as np
import tensorflow as tf


class VGG_text(object):  
    def __init__(self, num_classes, max_length, vocab_size, embedding_size, vgg19_npy_path=None, trainable=True,train_mode=None, dropout_keep_prob=0.5):
        self.data_dict = None 

        self.var_dict = {}
        self.trainable = trainable
        self.dropout_keep_prob = dropout_keep_prob
        
        self.input_x = tf.placeholder(tf.int32, [None, max_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y') 
        
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x) 
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) 
            
        #VGGnet_Bigram
        with tf.name_scope('VGGnet_Bigram'):
            filter_size = 2
            self.conv1_1 = self.conv_layer(self.embedded_chars_expanded,2,embedding_size,1, 
                                           max_length-filter_size+1, "conv1_1")
            self.pool1= tf.nn.max_pool(self.conv1_1, ksize=[1, max_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='SAME', name='pool1')
            
            self.conv2_1 = self.conv_layer(self.embedded_chars_expanded,3,embedding_size,1, 
                                           max_length-filter_size+1, "conv1_1")
            self.conv2_2 = self.conv_layer(self.conv2_1,2,embedding_size,max_length-filter_size +1, 
                                           max_length-filter_size +1, "conv1_2")
            self.pool2= tf.nn.max_pool(self.conv2_2, ksize=[1, max_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='SAME', name='pool1')
            
            self.conv3_1 = self.conv_layer(self.embedded_chars_expanded,4,embedding_size,1, 
                                           max_length-filter_size+1, "conv2_1")
            self.conv3_2 = self.conv_layer(self.conv3_1,3,embedding_size,max_length-filter_size +1, 
                                           max_length-filter_size+1, "conv2_2")
            self.conv3_3 = self.conv_layer(self.conv3_2,2,embedding_size,max_length-filter_size+1, 
                                           max_length-filter_size+1 , "conv2_3")
            self.pool3 = tf.nn.max_pool(self.conv3_3 , ksize=[1, max_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='SAME', name='pool2')


            num_filters_total = (max_length-filter_size+1)* (max_length *embedding_size) * 3
            
            total_pools = [self.pool1,self.pool2,self.pool3]    
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
            
            #self.data_dict = None
            
    def conv_layer(self, bottom, filter_size, embedding_size, in_channels, out_channels, layer_name):
        with tf.variable_scope(layer_name):
            filt, conv_biases = self.get_conv_var(filter_size, embedding_size, in_channels, out_channels, layer_name)

            conv = tf.nn.conv2d(bottom, filt, strides =[1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias, name=layer_name+'_relu' )

            return relu

    
    def fc_layer(self, bottom, in_size, out_size, layer_name):
        with tf.variable_scope(layer_name):
            weights, biases = self.get_fc_var(in_size, out_size, layer_name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
        
    def get_conv_var(self, filter_size, embedding_size,in_channels, out_channels, layer_name):
        initial_value = tf.truncated_normal([filter_size,embedding_size,in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, layer_name, 0, layer_name + "_filters")

        initial_value = tf.truncated_normal([out_channels], 0.0, 0.001)
        biases = self.get_var(initial_value, layer_name, 1, layer_name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, layer_name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, layer_name, 0, layer_name + "_weights")

        initial_value = tf.truncated_normal([out_size], 0.0, 0.001)
        biases = self.get_var(initial_value, layer_name, 1, layer_name + "_biases")

        return weights, biases

    def get_var(self, initial_value, layer_name, idx, var_name):
        value = initial_value        
        var = tf.Variable(value, name=var_name)
        return var

