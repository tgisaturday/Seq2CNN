import numpy as np
import tensorflow as tf


class text_Vgg19(object):  
    def __init__(self, sequence_length, num_classes, max_length, vocab_size, embedding_size, vgg19_npy_path=None, trainable=True,train_mode=None, dropout_keep_prob=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='utf8').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout_keep_prob = dropout_keep_prob
        
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y') 
        
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x) 
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) 
        
        assert self.embedded_chars_expanded.get_shape().as_list()[1:] == [max_length, embedding_size, 1]
                # Embedding layer
        #VGG19 Network
        with tf.name_scope('VGG19_Net'):

            self.conv1_1 = self.conv_layer(self.embedded_chars_expanded,1,64,"conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1,64,64,"conv1_2")
            self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
            
            self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
            self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool2')
            
            self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
            self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
            self.pool3 = tf.nn.max_pool(self.conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool3')

            self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
            self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
            self.pool4 = tf.nn.max_pool(self.conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool4')

            self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
            self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
            self.pool5 = tf.nn.max_pool(self.conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool5')
            
            self.fc6 = self.fc_layer(self.pool5,(max_length//32) *(embedding_size//32) * 512, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
            self.relu6 = tf.nn.relu(self.fc6)
            if train_mode is not None:
                self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout_keep_prob), lambda: self.relu6)
            elif self.trainable:
                self.relu6 = tf.nn.dropout(self.relu6, self.dropout_keep_prob)

            self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)
        
            if train_mode is not None:
                self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout_keep_prob), lambda: self.relu7)
            elif self.trainable:
                self.relu7 = tf.nn.dropout(self.relu7, self.dropout_keep_prob)

            self.fc8 = self.fc_layer(self.relu7, 4096, num_classes, "fc8")

            #self.prob = tf.nn.softmax(self.fc8, name="prob")
            self.prob = self.fc8
            self.data_dict = None
            # Final (unnormalized) scores and predictions
            with tf.name_scope('output'):
                W = tf.get_variable('W', shape=[4096, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.01, shape=[num_classes]), name='b')
                self.scores = self.fc8
                self.predictions = tf.argmax(self.fc8, 1, name='predictions')
        # Calculate mean cross-entropy loss
            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.scores)              # only named arguments accepted
                self.loss = tf.reduce_mean(losses)

        # Accuracy
            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
            with tf.name_scope('num_correct'):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
            
    def conv_layer(self, bottom, in_channels, out_channels, layer_name):
        with tf.variable_scope(layer_name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, layer_name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    
    def fc_layer(self, bottom, in_size, out_size, layer_name):
        with tf.variable_scope(layer_name):
            weights, biases = self.get_fc_var(in_size, out_size, layer_name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
        
    def get_conv_var(self, filter_size, in_channels, out_channels, layer_name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
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
        if self.data_dict is not None and layer_name in self.data_dict:
            value = self.data_dict[layer_name][idx]
        else:    
            value = initial_value
            
        var = tf.Variable(value, name=var_name)

        self.var_dict[(layer_name, idx)] = var
        
        assert var.get_shape() == initial_value.get_shape()

        return var
    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count