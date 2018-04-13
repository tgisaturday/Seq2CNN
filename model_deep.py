import numpy as np
import tensorflow as tf
import math
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

# weights initializers
he_normal = tf.keras.initializers.he_normal()
regularizer = tf.contrib.layers.l2_regularizer(1e-4)

class seq2CNN(object):  
    def __init__(self,embeddings, num_classes, max_summary_length, rnn_size, rnn_num_layers, vocab_to_int, num_filters, vocab_size, embedding_size, greedy,
                  depth=9, downsampling_type='maxpool', use_he_uniform=True, optional_shortcut=False):
        
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')        
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.text_length = tf.placeholder(tf.int32, (None,), name='text_length')
        self.summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        
        # Depth to No. Layers
        if depth == 9:
            num_layers = [2,2,2,2]
        elif depth == 17:
            num_layers = [4,4,4,4]
        elif depth == 29:
            num_layers = [10,10,4,4]
        elif depth == 49:
            num_layers = [16,16,10,6]
        else:
            raise ValueError('depth=%g is a not a valid setting!' % depth)
            
        
        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            enc_embed_input = tf.nn.embedding_lookup(embeddings, self.input_x)
            embedding_size = embedding_size
        #seq2seq layers
        with tf.name_scope('seq2seq'):
            batch_size = tf.reshape(self.batch_size, [])
            enc_output, enc_state = encoding_layer(rnn_size, self.text_length, rnn_num_layers, enc_embed_input, self.dropout_keep_prob)
            
            dec_input = process_encoding_input(self.targets, vocab_to_int, batch_size)
            dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
            training_logits  = decoding_layer(dec_embed_input,
                                                embeddings,
                                                enc_output,
                                                enc_state, 
                                                vocab_size, 
                                                self.text_length,
                                                self.summary_length,
                                                max_summary_length,
                                                rnn_size, 
                                                vocab_to_int, 
                                                self.dropout_keep_prob, 
                                                batch_size,
                                                rnn_num_layers,
                                                greedy)
            self.training_logits =tf.argmax(training_logits[0].rnn_output,axis=2,name='rnn_output',output_type=tf.int64)
        self.training_logits = tf.reshape(self.training_logits, [batch_size,max_summary_length])


    
        #VGGnet_Bigram
        with tf.name_scope('VDCNN'):
            self.decoder_output = tf.nn.embedding_lookup(embeddings, self.training_logits)
            self.cnn_input = tf.contrib.layers.batch_norm(self.decoder_output,center=True, scale=True,is_training=self.is_training)
            
            self.layers = []
            # Temp(First) Conv Layer
            with tf.variable_scope("temp_conv") as scope: 
                filter_shape = [3, embedding_size, 64]
                W = tf.get_variable(name='W_1', shape=filter_shape, 
                                    initializer=he_normal,
                                    regularizer=regularizer)
                inputs = tf.nn.conv1d(self.cnn_input, W, stride=1, padding="SAME")
                self.layers.append(inputs)
                
            # Conv Block 64
            for i in range(num_layers[0]):
                if i < num_layers[0] - 1 and optional_shortcut:
                    shortcut = self.layers[-1]
                else:
                    shortcut = None
                conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=64, is_training=self.is_training, name=str(i+1))
                self.layers.append(conv_block)
            pool1 = downsampling(self.layers[-1], downsampling_type=downsampling_type, name='pool1', optional_shortcut=optional_shortcut, shortcut=self.layers[-2])
            self.layers.append(pool1)
            
            # Conv Block 128
            for i in range(num_layers[1]):
                if i < num_layers[1] - 1 and optional_shortcut:
                    shortcut = self.layers[-1]
                else:
                    shortcut = None
                conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=128, is_training=self.is_training, name=str(i+1))
                self.layers.append(conv_block)
            pool2 = downsampling(self.layers[-1], downsampling_type=downsampling_type, name='pool2', optional_shortcut=optional_shortcut, shortcut=self.layers[-2])
            self.layers.append(pool2)

            # Conv Block 256
            for i in range(num_layers[2]):
                if i < num_layers[2] - 1 and optional_shortcut:
                    shortcut = self.layers[-1]
                else:
                    shortcut = None
                conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=256, is_training=self.is_training, name=str(i+1))
                self.layers.append(conv_block)
            pool3 = downsampling(self.layers[-1], downsampling_type=downsampling_type, name='pool3', optional_shortcut=optional_shortcut, shortcut=self.layers[-2])
            self.layers.append(pool3)

            # Conv Block 512
            for i in range(num_layers[3]):
                if i < num_layers[3] - 1 and optional_shortcut:
                    shortcut = self.layers[-1]
                else:
                    shortcut = None
                conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=512, is_training=self.is_training, name=str(i+1))
                self.layers.append(conv_block)

            # Extract 8 most features as mentioned in paper
            self.k_pooled = tf.nn.top_k(tf.transpose(self.layers[-1], [0,2,1]), k=8, name='k_pool', sorted=False)[0]
            print("8-maxpooling:", self.k_pooled.get_shape())
            self.flatten = tf.reshape(self.k_pooled, (-1, 512*8))

            # fc1
            with tf.variable_scope('fc1'):
                w = tf.get_variable('w', [self.flatten.get_shape()[1], 2048], initializer=he_normal,
                    regularizer=regularizer)
                b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
                out = tf.matmul(self.flatten, w) + b
                self.fc1 = tf.nn.relu(out)

            # fc2
            with tf.variable_scope('fc2'):
                w = tf.get_variable('w', [self.fc1.get_shape()[1], 2048], initializer=he_normal,
                    regularizer=regularizer)
                b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
                out = tf.matmul(self.fc1, w) + b
                self.fc2 = tf.nn.relu(out)

        # fc3
            with tf.variable_scope('fc3'):
                w = tf.get_variable('w', [self.fc2.get_shape()[1], num_classes], initializer=he_normal,
                    regularizer=regularizer)
                b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(1.0))
                self.fc3 = tf.matmul(self.fc2, w) + b

                
        with tf.name_scope('output'):
            self.scores = self.fc3
            self.predictions = tf.argmax(self.scores, 1, name='predictions')
                
        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            cnn_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,logits=self.scores)
            masks = tf.sequence_mask(self.summary_length, max_summary_length, dtype=tf.float32, name='masks')
            seq_loss = tf.contrib.seq2seq.sequence_loss(training_logits[0].rnn_output,self.targets,masks)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(cnn_loss) + sum(regularization_losses)
            self.seq_loss = seq_loss
        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')

def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['GO']), ending], 1)

    return dec_input 

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    '''Create the encoding layer'''
    
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
       
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            #cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
            #cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_size,layer_norm=True,dropout_keep_prob= keep_prob)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            #cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_size,layer_norm=True,dropout_keep_prob= keep_prob)
            #cell_bw = tf.contrib.rnn.GRUCell(rnn_size)                                                       
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,input_keep_prob = keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                    cell_bw, 
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output,2)
    
    return enc_output, enc_state

def training_decoding_layer(embeddings, dec_embed_input, summary_length, start_token, end_token, dec_cell, initial_state, output_layer, vocab_size, max_summary_length, batch_size, greedy):
    '''Create the training logits'''
    if greedy:
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    
        training_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)
    else:
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=summary_length,
                                                            time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer) 

    training_logits = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                           output_time_major=False,
                                                           impute_finished=True,
                                                          maximum_iterations=max_summary_length)
    return training_logits



def decoding_layer(dec_embed_input,embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, greedy):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            #lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_size,layer_norm=True,dropout_keep_prob= keep_prob)
            #lstm = tf.contrib.rnn.GRUCell(rnn_size)             
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                     input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  text_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                   attention_mechanism = attn_mech,
                                                   attention_layer_size = rnn_size,
                                                   name='Attention_Wrapper')
    
    initial_state =dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    initial_state = initial_state.clone(cell_state=enc_state[0])

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(embeddings,dec_embed_input,
                                                  summary_length,                                                                                                                                           vocab_to_int['GO'], 
                                                  vocab_to_int['EOS'],
                                                  dec_cell, 
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size, 
                                                  max_summary_length,
                                                  batch_size,
                                                  greedy)  

    return training_logits

def Convolutional_Block(inputs, shortcut, num_filters, name, is_training):

    with tf.variable_scope("conv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            with tf.variable_scope("conv1d_%s" % str(i)):
                filter_shape = [3, inputs.get_shape()[2], num_filters]
                W = tf.get_variable(name='W', shape=filter_shape, 
                    initializer=he_normal,
                    regularizer=regularizer)
                inputs = tf.nn.conv1d(inputs, W, stride=1, padding="SAME")
                inputs = tf.contrib.layers.batch_norm(inputs,center=True, scale=True,is_training=is_training)
                inputs = tf.nn.relu(inputs)

    if shortcut is not None:
        return inputs + shortcut
    return inputs

def downsampling(inputs, downsampling_type, name, optional_shortcut=False, shortcut=None):
    # k-maxpooling
    if downsampling_type=='k-maxpool':
        k = math.ceil(int(inputs.get_shape()[1]) / 2)
        pool = tf.nn.top_k(tf.transpose(inputs, [0,2,1]), k=k, name=name, sorted=False)[0]
        pool = tf.transpose(pool, [0,2,1])
    # Linear
    elif downsampling_type=='linear':
        pool = tf.layers.conv1d(inputs=inputs, filters=inputs.get_shape()[2], kernel_size=3,
                            strides=2, padding='same', use_bias=False)
    # Maxpooling
    else:
        pool = tf.layers.max_pooling1d(inputs=inputs, pool_size=3, strides=2, padding='same', name=name)
    if optional_shortcut:
        shortcut = tf.layers.conv1d(inputs=shortcut, filters=shortcut.get_shape()[2], kernel_size=1,
                            strides=2, padding='same', use_bias=False)
        print("-"*5)
        print("Optional Shortcut:", shortcut.get_shape())
        print("-"*5)
        pool += shortcut
    pool = fixed_padding(inputs=pool)
    return tf.layers.conv1d(inputs=pool, filters=pool.get_shape()[2]*2, kernel_size=1,
                            strides=1, padding='valid', use_bias=False)

def fixed_padding(inputs, kernel_size=3):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


#------------------not used-------------------------------------------
def get_batches(summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))
        
        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))
        
        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))
        
        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths
        
def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer, max_summary_length, batch_size):
    '''Create the inference logits'''
    
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)
                
    inference_decoder, _, _  = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)
                
    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=max_summary_length)
    
    return inference_logits
