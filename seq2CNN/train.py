import os
import sys
import json
import time
import logging
import data_helper
import numpy as np
import tensorflow as tf
import math
from model import seq2CNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)  

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1
                
def pad_sentence_batch(vocab_to_int,sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    result = []
    for sentence in sentence_batch:
        result.append(sentence + [vocab_to_int['PAD']] * (max_sentence - len(sentence)))
    return result

def convert_to_ints(text,vocab_to_int, word_count, unk_count, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["UNK"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["EOS"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count
def exponential_lambda_decay(seq_lambda, global_step, decay_steps, decay_rate, staircase=False):
    global_step = float(global_step)
    decay_steps = float(decay_steps)
    decay_rate = float(decay_rate)
    p = global_step / decay_steps
    if staircase:
        p = math.floor(p)
    return seq_lambda * math.pow(decay_rate, p)
def train_cnn(dataset_name):
    """Step 0: load sentences, labels, and training parameters"""
    dataset = '../dataset/'+dataset_name+'_csv/train.csv'
    testset = '../dataset/'+dataset_name+'_csv/test.csv'
    parameter_file = "./parameters.json"
    params = json.loads(open(parameter_file).read())
    learning_rate = params['learning_rate']
    filter_sizes = list(int(x) for x in params['filter_sizes'].split(','))
    if params['enable_max_len'] == 1:
        enable_max = True
    else:
        enable_max = False
    if params['use_gru'] == 1:
        use_gru = True
    else:
        use_gru = False
    if params['rnn_layer_norm'] == 1:
        rnn_layer_norm= True
    else:
        rnn_layer_norm = False
    if params['fc_layer_norm'] == 1:
        fc_layer_norm= True
    else:
        fc_layer_norm = False
    if params['temp_norm'] == 1:
        temp_norm= True
    else:
        temp_norm = False
    if params['watch_rnn_output'] == 1:
        watch_rnn_output = True
    else:
        watch_rnn_output = False
    if params['independent_train'] == 1:
        independent_train = True
    else:
        independent_train = False
    if params['is_simple'] == 1:
        is_simple = True
    else:
        is_simple = False        
    x_raw, y_raw, target_raw, df, labels = data_helper.load_data_and_labels(dataset,dataset_name,params['max_length'],params['max_summary_length'],enable_max,True)
    x_test_raw, y_test_raw, target_test_raw, df_test, labels_test = data_helper.load_data_and_labels(testset,dataset_name,params['max_length'],params['max_summary_length'],enable_max,False)
    word_counts = {}
    count_words(word_counts, x_raw)        
    logging.info("Size of Vocabulary: {}".format(len(word_counts)))

    """Step 1: pad each sentence to the same length and map each word to an id"""
    max_document_length = max([len(x.split(' ')) for x in x_raw])
    min_document_length = min([len(x.split(' ')) for x in x_raw])
    logging.info('The maximum length of all sentences: {}'.format(max_document_length))
    logging.info('The minimum length of all sentences: {}'.format(min_document_length))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,
                                                              min_frequency=params['min_frequency'])
    vocab_processor.fit_transform(x_raw)
    vocab_to_int = vocab_processor.vocabulary_._mapping
    
    # Special tokens that will be added to our vocab
    codes = ["UNK","PAD","EOS","GO"]   

    # Add codes to vocab
    for code in codes:
        vocab_to_int[code] = len(vocab_to_int)

    # Dictionary to convert integers to words
    int_to_vocab = {}
    for word, value in vocab_to_int.items():
        int_to_vocab[value] = word
    usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

    logging.info("Total number of words: {}".format(len(word_counts)))
    logging.info("Number of words we will use: {}".format(len(vocab_to_int)))
    logging.info("Percent of words we will use: {0:.2f}%".format(usage_ratio))
    

    # Apply convert_to_ints to clean_summaries and clean_texts
    word_count = 0
    unk_count = 0
    int_summaries, word_count, unk_count = convert_to_ints(target_raw,vocab_to_int, word_count, unk_count)
    int_texts, word_count, unk_count = convert_to_ints(x_raw,vocab_to_int, word_count, unk_count, eos=True)
    int_test_summaries, word_count, unk_count = convert_to_ints(target_test_raw,vocab_to_int, word_count, unk_count)
    int_test_texts, word_count, unk_count = convert_to_ints(x_test_raw,vocab_to_int, word_count, unk_count, eos=True)    
    unk_percent = round(unk_count/word_count,4)*100

    logging.info("Total number of words in texts: {}".format(word_count))
    logging.info("Total number of UNKs in  texts: {}".format(unk_count))
    logging.info("Percent of words that are UNK: {0:.2f}%".format(unk_percent))
    
    """Step 1: pad each sentence to the same length and map each word to an id"""

    x_int = pad_sentence_batch(vocab_to_int,int_texts)
    target_int = pad_sentence_batch(vocab_to_int,int_summaries)
    x_test_int = pad_sentence_batch(vocab_to_int,int_test_texts)
    target_test_int = pad_sentence_batch(vocab_to_int,int_test_summaries)
    x = np.array(x_int)
    y = np.array(y_raw)
    x_test = np.array(x_test_int)
    y_test = np.array(y_test_raw)
    
    target = np.array(target_int)
    target_test = np.array(target_test_int)    
    t = np.array(list(len(x) for x in x_int))
    t_test = np.array(list(len(x) for x in x_test_int))
    s = np.array(list(params['max_summary_length'] for x in x_int))
    s_test = np.array(list(params['max_summary_length'] for x in x_test_int))



    """Step 3: shuffle the train set and split the train set into train and dev sets"""
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    target_shuffled = target[shuffle_indices]
    t_shuffled = t[shuffle_indices]
    s_shuffled = s[shuffle_indices]
    x_train, x_dev, y_train, y_dev,target_train, target_dev, t_train, t_dev,s_train, s_dev = train_test_split(x_shuffled, y_shuffled,target_shuffled, t_shuffled,s_shuffled, test_size=0.1)

    """Step 4: save the labels into labels.json since predict.py needs it"""
    with open('./labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))
    logging.info('target_train: {}, target_dev: {}, target_test: {}'.format(len(target_train), len(target_dev), len(target_test)))
    logging.info('t_train: {}, t_dev: {}, t_test: {}'.format(len(t_train), len(t_dev), len(t_test)))
    logging.info('s_train: {}, s_dev: {}, s_test: {}'.format(len(s_train), len(s_dev), len(s_test)))

    """Step 5: build a graph and cnn object"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            cnn = seq2CNN(
                num_classes=y_train.shape[1],
                filter_sizes=filter_sizes,
                max_summary_length=params['max_summary_length'],
                rnn_size=params['rnn_size'],
                rnn_num_layers=params['rnn_num_layers'],
                vocab_to_int = vocab_to_int,
                num_filters=params['num_filters'],
                vocab_size=len(vocab_to_int),
                embedding_size=params['embedding_dim']
                )
            global_step = tf.Variable(0, name="global_step", trainable=False)            
            num_batches_per_epoch = int((len(x_train)-1)/params['batch_size']) + 1
            epsilon=params['epsilon']

            learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step,params['num_epochs']*num_batches_per_epoch, 0.95, staircase=True)

            optimizer = tf.train.AdamOptimizer(learning_rate,epsilon)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            gradients, variables = zip(*optimizer.compute_gradients(cnn.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, dataset_name + "_" + timestamp))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            #for tensorboard
            train_writer = tf.summary.FileWriter('/home/tgisaturday/Workspace/Taehoon/VGG_text_cnn/seq2CNN'+'/graphs/train/'+dataset_name+'_'+timestamp,sess.graph)
            test_writer = tf.summary.FileWriter('/home/tgisaturday/Workspace/Taehoon/VGG_text_cnn/seq2CNN'+'/graphs/test/'+dataset_name+'_'+timestamp) 
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())
            seq_lambda = params['seq_lambda']
            # One training step: train the model with one batch
            def train_step(x_batch, y_batch,target_batch,t_batch,s_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.targets: target_batch,
                    cnn.text_length: t_batch,
                    cnn.summary_length: s_batch,
                    cnn.batch_size: len(x_batch),
                    cnn.dropout_keep_prob: params['dropout_keep_prob'],
                    cnn.seq_lambda: seq_lambda,
                    cnn.is_training: True}
                summary, _, logits, step, loss,seq_loss, cnn_loss, acc = sess.run([cnn.merged,train_op,cnn.training_logits, global_step, cnn.loss, cnn.seq_loss, cnn.cnn_loss, cnn.accuracy], feed_dict)
                current_step = tf.train.global_step(sess, global_step)
                train_writer.add_summary(summary,current_step)
                return loss, seq_loss, cnn_loss, acc, logits

            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch, y_batch,target_batch, t_batch,s_batch):
                feed_dict = {
                    cnn.input_x: x_batch, 
                    cnn.input_y: y_batch,
                    cnn.targets: target_batch,
                    cnn.text_length: t_batch,
                    cnn.summary_length: s_batch,
                    cnn.batch_size: len(x_batch),
                    cnn.dropout_keep_prob: 1.0,
                    cnn.seq_lambda: seq_lambda,
                    cnn.is_training: False}
                summary, step, loss, seq_loss, acc, num_correct,examples = sess.run([cnn.merged,global_step, cnn.loss, cnn.seq_loss, cnn.accuracy, cnn.num_correct,cnn.inference_logits],feed_dict)
                if watch_rnn_output == True:
                    pad = vocab_to_int['PAD']
                    result =  " ".join([int_to_vocab[j] for j in examples[0] if j != pad])
                    logging.info('{}'.format(result))
                current_step = tf.train.global_step(sess, global_step)
                test_writer.add_summary(summary,current_step) 
                return num_correct

            # Save the word_to_id map since predict.py needs it
            vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))

            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = data_helper.batch_iter(list(zip(x_train, y_train,target_train,t_train,s_train)), params['batch_size'],
                                                   params['num_epochs'])
            best_accuracy, best_at_step = 0, 0

            """Step 6: train the cnn model with x_train and y_train (batch by batch)"""
            for train_batch in train_batches:
                x_train_batch, y_train_batch,target_train_batch, t_train_batch,s_train_batch = zip(*train_batch)
                current_step = tf.train.global_step(sess, global_step)
                train_loss, train_seq_loss, train_cnn_loss,train_acc,examples = train_step(x_train_batch, y_train_batch,target_train_batch,t_train_batch,s_train_batch)

                """Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
                    logging.critical('step: {} accuracy: {} loss: {} seq_loss: {} cnn_loss: {}'.format(current_step, train_acc, train_loss, train_seq_loss,train_cnn_loss))
                    dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev,target_dev,t_dev,s_dev)), params['batch_size'], 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch,target_dev_batch, t_dev_batch,s_dev_batch = zip(*dev_batch)
                        num_dev_correct = dev_step(x_dev_batch, y_dev_batch,target_dev_batch,t_dev_batch,s_dev_batch)
                        total_dev_correct += num_dev_correct

                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

                    """Step 6.2: save the model if it is the best based on accuracy on dev set"""
                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model at {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy is {} at step {}'.format(best_accuracy, best_at_step))
                seq_lambda = exponential_lambda_decay(params['seq_lambda'], current_step,params['num_epochs']*num_batches_per_epoch, 0.95, staircase=True)
            """Step 7: predict x_test (batch by batch)"""
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test,target_test,t_test,s_test)), params['batch_size'], 1)
            total_test_correct = 0
            watch_rnn_output = True
            start = time.time()
            for test_batch in test_batches:
                x_test_batch, y_test_batch,target_test_batch, t_test_batch,s_test_batch = zip(*test_batch)
                num_test_correct = dev_step(x_test_batch, y_test_batch,target_test_batch,t_test_batch,s_test_batch)
                total_test_correct += num_test_correct
            path = saver.save(sess, checkpoint_prefix)
            test_accuracy = float(total_test_correct) / len(y_test)
            logging.critical("\nExecution time for testing = {0:.6f}".format(time.time() - start))            
            logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
            logging.critical('The training is complete')


if __name__ == '__main__':
    # python3 train.py ./data/consumer_complaints.csv.zip ./parameters.json
    train_cnn(sys.argv[1])
