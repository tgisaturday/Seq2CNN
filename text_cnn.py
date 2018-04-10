import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import ntm.ntm_cell as ntm_cell

class seq2CNN(object):  
    def __init__(self,num_classes, max_summary_length, rnn_size, rnn_num_layers, vocab_to_int, num_filters, vocab_size, embedding_size,greedy):
        
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')        
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.text_length = tf.placeholder(tf.int32, (None,), name='text_length')
        self.summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        
        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            enc_embed_input = tf.nn.embedding_lookup(embeddings, self.input_x)
            embedding_size = embedding_size
            #self.decoder_output_expanded = tf.expand_dims(self.enc_embed_input, -1) 
        #seq2seq layers
        with tf.name_scope('seq2seq'):
            #embeddings = word_embedding_matrix
            #enc_embed_input = tf.nn.embedding_lookup(embeddings, self.input_x)
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
        with tf.name_scope('VGGnet_Bigram'):
            W = embeddings
            self.decoder_output = tf.nn.embedding_lookup(W, self.training_logits)
            self.decoder_output_expanded = tf.expand_dims(self.decoder_output, -1) 
            filter_sizes=[3,4,5]
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                    conv = tf.nn.conv2d(self.decoder_output_expanded, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(h, ksize=[1, max_summary_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name='pool')
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            
            #filter_size = 3
            #filter_shape = [3, embedding_size, 1, num_filters*2]
            #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
           # b = tf.Variable(tf.constant(0.1, shape=[num_filters*2]), name='b')
            #conv1_1 = tf.nn.conv2d(self.decoder_output_expanded, W, strides=[1, 1, 1, 1], padding='SAME', name='conv1_1')
            #h1_1 = tf.nn.leaky_relu(tf.nn.bias_add(conv1_1, b), alpha=0.1,  name='relu1_1')
            
           # pool1= tf.nn.max_pool(h1_1, ksize=[1, num_filters*2, 1, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool1')
            
            #filter_shape = [3, embedding_size, 1, num_filters]
            #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            #b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            #conv2_1 = tf.nn.conv2d(self.decoder_output_expanded, W, strides=[1, 1, 1, 1], padding='SAME', name='conv2_1')
            #h2_1 = tf.nn.leaky_relu(tf.nn.bias_add(conv2_1, b), alpha=0.1, name='relu2_1')

            #filter_shape = [3, embedding_size, num_filters, num_filters*2]
            #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            #b = tf.Variable(tf.constant(0.1, shape=[num_filters*2]), name='b')
            #conv2_2 = tf.nn.conv2d(h2_1, W, strides=[1, 1, 1, 1], padding='SAME', name='conv2_2')
            #h2_2 = tf.nn.leaky_relu(tf.nn.bias_add(conv2_2, b), alpha=0.1,  name='relu2_2')
            
            #pool2= tf.nn.max_pool(h2_2, ksize=[1,num_filters*2, 1, 1], strides=[1, 1, 1, 1],padding='SAME', name='pool2')
            
            #filter_shape = [3, embedding_size, 1, num_filters]
           #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
           #b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            #conv3_1 = tf.nn.conv2d(self.decoder_output_expanded, W, strides=[1, 1, 1, 1], padding='SAME', name='conv3_1')
           # h3_1 = tf.nn.leaky_relu(tf.nn.bias_add(conv3_1, b), alpha=0.1,  name='relu3_1')
            
            #filter_shape = [3, embedding_size, num_filters, num_filters*2]
            #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            #b = tf.Variable(tf.constant(0.1, shape=[num_filters*2]), name='b')
            #conv3_2 = tf.nn.conv2d(h3_1, W, strides=[1, 1, 1, 1], padding='SAME', name='conv3_2')
            #h3_2 = tf.nn.leaky_relu(tf.nn.bias_add(conv3_2, b),  alpha=0.1, name='relu3_2')
            
            #filter_shape = [3, embedding_size, num_filters*2, num_filters*2]
            #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
           # b = tf.Variable(tf.constant(0.1, shape=[num_filters*2]), name='b')
            #conv3_3 = tf.nn.conv2d(h3_2, W, strides=[1, 1, 1, 1], padding='SAME', name='conv3_3')
            #h3_3 = tf.nn.leaky_relu(tf.nn.bias_add(conv3_3, b),  alpha=0.1, name='relu3_3')
            
            #pool3 = tf.nn.max_pool(h3_3 , ksize=[1,num_filters*2, 1, 1], strides=[1, 1, 1, 1],padding='SAME', name='pool3')

            #num_filters_total = num_filters * (max_summary_length * embedding_size) 
            
            #total_pools = [pool1,pool2,pool3]    
            #self.pool_h = tf.concat(total_pools, 3)
            #self.pool_h_flat = tf.reshape(self.pool_h, [-1, num_filters_total])
            #self.h_pool_flat = tf.reshape(pool1, [-1, num_filters_total]) 
            
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
                
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
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
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

def training_decoding_layer(embeddings, dec_embed_input, summary_length, start_token, end_token, dec_cell, initial_state, output_layer, vocab_size, max_summary_length, batch_size,greedy):
    '''Create the training logits'''
    if greedy:
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    
        training_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,start_tokens,end_token)
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



def decoding_layer(dec_embed_input,embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers,greedy):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    
    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
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
                                                  summary_length,                                                                                     vocab_to_int['GO'], 
                                                  vocab_to_int['EOS'],
                                                  dec_cell, 
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size, 
                                                  max_summary_length,
                                                  batch_size,greedy)  

    return training_logits

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