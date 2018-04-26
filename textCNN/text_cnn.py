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
        self.h_outputs=[]
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [3, embedding_size, 1, 32]
                W = tf.get_variable(name='W', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
                #Apply nonlinearity

                h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
                h = tf.nn.relu(h, name='relu')
                self.h_outputs.append(h)
                
        h_total = tf.concat(self.h_outputs, 3)        
        #64    
        filter_shape = [3, 1, len(filter_sizes)*32, 64]
        W_2_1 = tf.get_variable(name='W_2_1', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        conv = tf.nn.conv2d(h_total, W_2_1, strides=[1, 2, 1, 1], padding='SAME', name='conv')
            
        h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        h = tf.nn.relu(h, name='relu')

        filter_shape = [3, 1, 64, 64]
        W_2_2 = tf.get_variable(name='W_2_2', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        conv = tf.nn.conv2d(h, W_2_2, strides=[1, 2, 1, 1], padding='SAME', name='conv')

        h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        h = tf.nn.relu(h, name='relu')
                      
        pooled = tf.nn.max_pool(h, ksize=[1, 2 , 1, 1], strides=[1, 2, 1, 1], padding='SAME', name='pool') 
                
        #128
        filter_shape = [3, 1, 64, 128]
        W_3_1 = tf.get_variable(name='W_3_1', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        conv = tf.nn.conv2d(pooled, W_3_1, strides=[1, 2, 1, 1], padding='SAME', name='conv')

        h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        h = tf.nn.relu(h, name='relu')

        filter_shape = [3, 1, 128, 128]
        W_3_2 = tf.get_variable(name='W_3_2', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        conv = tf.nn.conv2d(h, W_3_2, strides=[1, 2, 1, 1], padding='SAME', name='conv')

        h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        h = tf.nn.relu(h, name='relu')

        # Maxpooling over the outputs                        
        pooled = tf.nn.max_pool(h, ksize=[1, 2 , 1, 1], strides=[1, 2, 1, 1],padding='SAME', name='pool')
            
        #256
        filter_shape = [3, 1, 128, 256]
        W_4_1 = tf.get_variable(name='W_4_1', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        conv = tf.nn.conv2d(pooled, W_4_1, strides=[1, 2, 1, 1], padding='SAME', name='conv')

        h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        h = tf.nn.relu(h, name='relu')

        filter_shape = [3, 1, 256, 256]
        W_4_2 = tf.get_variable(name='W_4_2', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        conv = tf.nn.conv2d(h, W_4_2, strides=[1, 2, 1, 1], padding='SAME', name='conv')

        h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        h = tf.nn.relu(h, name='relu')
            
        filter_shape = [3, 1, 256, 256]
        W_4_3 = tf.get_variable(name='W_4_3', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        conv = tf.nn.conv2d(h, W_4_3, strides=[1, 2, 1, 1], padding='SAME', name='conv')

        h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        h = tf.nn.relu(h, name='relu')
            
        #filter_shape = [3, 1, 256, 256]
        #W_4_4 = tf.get_variable(name='W_4_4', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        #conv = tf.nn.conv2d(h, W_4_4, strides=[1, 2, 1, 1], padding='SAME', name='conv')

        #h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        #h = tf.nn.relu(h, name='relu')

        # Maxpooling over the outputs                        
        pooled = tf.nn.max_pool(h, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1],padding='SAME', name='pool') 
            
        #512
        filter_shape = [3, 1, 256, 512]
        W_5_1 = tf.get_variable(name='W_5_1', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        conv = tf.nn.conv2d(pooled, W_5_1, strides=[1, 2, 1, 1], padding='SAME', name='conv')

        h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        h = tf.nn.relu(h, name='relu')

        filter_shape = [3, 1, 512, 512]
        W_5_2 = tf.get_variable(name='W_5_2', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        conv = tf.nn.conv2d(h, W_5_2, strides=[1, 2, 1, 1], padding='SAME', name='conv')

        h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        h = tf.nn.relu(h, name='relu')
            
        filter_shape = [3, 1, 512, 512]
        W_5_3 = tf.get_variable(name='W_5_3', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        conv = tf.nn.conv2d(h, W_5_3, strides=[1, 2, 1, 1], padding='SAME', name='conv')

        h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        h = tf.nn.relu(h, name='relu')
            
        #filter_shape = [3, 1, 512, 512]
        #W_5_4 = tf.get_variable(name='W_5_4', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
        #conv = tf.nn.conv2d(h, W_5_4, strides=[1, 2, 1, 1], padding='SAME', name='conv')

        #h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
        #h = tf.nn.relu(h, name='relu')

        # Maxpooling over the outputs                        
        pooled = tf.nn.max_pool(h, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1],padding='SAME', name='pool')            
            

        # Combine all the pooled features
        with tf.variable_scope('fc-dropout-6'):
            h_pool = pooled
            h_pool_flat = tf.reshape(h_pool, [-1, h_pool.get_shape()[2]*512])
            W_fc6 = tf.get_variable('W_fc6', shape=[h_pool.get_shape()[2]*512, 512],
                                initializer=he_normal,regularizer = regularizer)
            b_fc6 = tf.get_variable('b_fc6', [512], initializer=tf.constant_initializer(0.1))
            fc6 =  tf.nn.xw_plus_b(h_pool_flat, W_fc6, b_fc6, name='fc6') 
            #fc6 = tf.contrib.layers.batch_norm(fc6,center=True, scale=True,is_training=self.is_training)                
            relu_fc6 =tf.nn.relu(fc6)
            self.fc6 = tf.nn.dropout(relu_fc6, self.dropout_keep_prob)
        
        with tf.variable_scope('fc-dropout-7'):
            W_fc7 = tf.get_variable('W_fc7', shape=[512, 512],
                                initializer=he_normal,regularizer = regularizer)
            b_fc7 = tf.get_variable('b_fc7', [512], initializer=tf.constant_initializer(0.1))
            fc7 =  tf.nn.xw_plus_b(self.fc6, W_fc7, b_fc7, name='fc7')

            #fc7 = tf.contrib.layers.batch_norm(fc7,center=True, scale=True,is_training=self.is_training)                
            relu_fc7 =tf.nn.relu(fc7)
            self.fc7 =tf.nn.dropout(relu_fc7, self.dropout_keep_prob)
        
        with tf.variable_scope('fc-dropout-8'):            
            W_fc8 = tf.get_variable('W_fc8', shape=[512, num_classes],
                                    initializer=initializer,regularizer = regularizer)
            b_fc8 = tf.get_variable('b_fc8', [num_classes], initializer=tf.constant_initializer(0.1))
            self.scores = tf.nn.xw_plus_b(self.fc7, W_fc8, b_fc8, name='fc8')
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