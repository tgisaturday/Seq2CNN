import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

initializer = tf.contrib.layers.xavier_initializer()
he_normal = tf.keras.initializers.he_normal()
rand_uniform = tf.random_uniform_initializer(-1,1,seed=2)
regularizer = tf.contrib.layers.l2_regularizer(1e-3)

class seq2CNN(object):  
    def __init__(self,num_classes,filter_sizes, max_summary_length, rnn_size, rnn_num_layers, vocab_to_int, num_filters, vocab_size, embedding_size, seq_ratio):
        
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')        
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.text_length = tf.placeholder(tf.int32, (None,), name='text_length')
        self.summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        
        
        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            embeddings = tf.get_variable(name='embedding_W', shape=[vocab_size, embedding_size],initializer=rand_uniform)
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
                                                rnn_num_layers)
            self.training_logits =tf.argmax(training_logits[0].rnn_output,axis=2,name='rnn_output',output_type=tf.int64)
        self.training_logits = tf.reshape(self.training_logits, [batch_size,max_summary_length])


    
        #VGGnet_Bigram
        with tf.name_scope('textCNN'):
            self.decoder_output = tf.nn.embedding_lookup(embeddings, self.training_logits)
            self.decoder_output_expanded = tf.expand_dims(self.decoder_output, -1)

            self.cnn_input = tf.contrib.layers.batch_norm(self.decoder_output_expanded,center=True, scale=True,is_training=self.is_training)

            
            self.pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope('conv-maxpool-%s' % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, 32]
                    W = tf.get_variable(name='W', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
                    conv = tf.nn.conv2d(self.cnn_input, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
                    #Apply nonlinearity

                    h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
                    h = tf.nn.relu(h, name='relu')
                    #self.h_outputs.append(h)     
                    #64    
                    filter_shape = [3, 1, 32, 32]
                    W_2_1 = tf.get_variable(name='W_2_1', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
                    conv = tf.nn.conv2d(h, W_2_1, strides=[1, 1, 1, 1], padding='SAME', name='conv')
            
                    h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
                    h = tf.nn.relu(h, name='relu')

                    #filter_shape = [3, 1, 64, 64]
                    #W_2_2 = tf.get_variable(name='W_2_2', shape=filter_shape,initializer=he_normal,regularizer=regularizer)
                    #conv = tf.nn.conv2d(h, W_2_2, strides=[1, 2, 1, 1], padding='SAME', name='conv')

                    #h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
                    #h = tf.nn.relu(h, name='relu')
                      
                    pooled = tf.nn.max_pool(h, ksize=[1, 2 , 1, 1], strides=[1, 2, 1, 1], padding='SAME', name='pool') 
                    self.pooled_outputs.append(pooled)
            
        # Combine all the pooled features
            with tf.variable_scope('fc-dropout-6'):
                h_pool = tf.concat(self.pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, h_pool.get_shape()[2]* h_pool.get_shape()[1]* 32*len(filter_sizes)])
                W_fc6 = tf.get_variable('W_fc6', shape=[h_pool.get_shape()[2]* h_pool.get_shape()[1]* 32*len(filter_sizes), 32*len(filter_sizes)],
                                    initializer=he_normal,regularizer = regularizer)
                b_fc6 = tf.get_variable('b_fc6', [32*len(filter_sizes)], initializer=tf.constant_initializer(0.1))
                fc6 =  tf.nn.xw_plus_b(h_pool_flat, W_fc6, b_fc6, name='fc6') 
                #fc6 = tf.contrib.layers.batch_norm(fc6,center=True, scale=True,is_training=self.is_training)                
                relu_fc6 =tf.nn.relu(fc6)
                self.fc6 = tf.nn.dropout(relu_fc6, self.dropout_keep_prob)
        
            with tf.variable_scope('fc-dropout-7'):
                W_fc7 = tf.get_variable('W_fc7', shape=[32*len(filter_sizes),  32*len(filter_sizes)],
                                        initializer=he_normal,regularizer = regularizer)
                b_fc7 = tf.get_variable('b_fc7', [32*len(filter_sizes)], initializer=tf.constant_initializer(0.1))
                fc7 =  tf.nn.xw_plus_b(self.fc6, W_fc7, b_fc7, name='fc7')

                #fc7 = tf.contrib.layers.batch_norm(fc7,center=True, scale=True,is_training=self.is_training)                
                relu_fc7 =tf.nn.relu(fc7)
                self.fc7 =tf.nn.dropout(relu_fc7, self.dropout_keep_prob)
        
            with tf.variable_scope('fc-dropout-8'):            
                W_fc8 = tf.get_variable('W_fc8', shape=[32*len(filter_sizes), num_classes],
                                        initializer=initializer,regularizer = regularizer)
                b_fc8 = tf.get_variable('b_fc8', [num_classes], initializer=tf.constant_initializer(0.1))
                self.scores = tf.nn.xw_plus_b(self.fc7, W_fc8, b_fc8, name='fc8')
                self.predictions = tf.argmax(self.scores, 1, name='predictions')

               
        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):            
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            cnn_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,logits=self.scores)
            masks = tf.sequence_mask(self.summary_length, max_summary_length, dtype=tf.float32, name='masks')
            seq_loss = tf.contrib.seq2seq.sequence_loss(training_logits[0].rnn_output,self.targets,masks)
            self.loss = tf.reduce_mean(cnn_loss) + seq_ratio*seq_loss + tf.reduce_sum(regularization_losses)
            self.seq_loss = seq_loss
            self.cnn_loss = tf.reduce_mean(cnn_loss)+ tf.reduce_sum(regularization_losses)
            tf.summary.scalar('seq_loss',self.seq_loss)
            tf.summary.scalar('cnn_loss',self.cnn_loss)
        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
            tf.summary.scalar('accuracy',self.accuracy)
        with tf.name_scope('num_correct'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
        self.merged = tf.summary.merge_all()    
def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['GO']), ending], 1)

    return dec_input 

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    '''Create the encoding layer'''
    
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):

            #cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_size,layer_norm=True,dropout_keep_prob= keep_prob)
            #cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_size,layer_norm=True,dropout_keep_prob= keep_prob)

            cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

            cell_bw = tf.contrib.rnn.GRUCell(rnn_size)   
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,input_keep_prob = keep_prob)

            #cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            #cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

            #cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))   
            #cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,input_keep_prob = keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                    cell_bw, 
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output,2)
    
    return enc_output, enc_state

def training_decoding_layer(embeddings, dec_embed_input, summary_length, start_token, end_token, dec_cell, initial_state, output_layer, vocab_size, max_summary_length, batch_size):

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



def decoding_layer(dec_embed_input,embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):

            #dec_cell =tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_size,layer_norm=True,dropout_keep_prob= keep_prob)

            lstm = tf.contrib.rnn.GRUCell(rnn_size)
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)

            #lstm = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            #dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    
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
                                                  batch_size)  

    return training_logits


