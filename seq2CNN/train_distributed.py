import os
import sys
import json
import time
import logging
import data_helper_distributed as data_helper
import numpy as np
import tensorflow as tf
from model_distributed import seq2CNN
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

def train_cnn(dataset_name):
    """Step 0: load sentences, labels, and training parameters"""
    dataset = '../dataset/'+dataset_name+'_csv/train.csv'
    testset = '../dataset/'+dataset_name+'_csv/test.csv'
    parameter_file = "./parameters.json"
    params = json.loads(open(parameter_file).read())
    learning_rate = params['learning_rate']
    if params['enable_max_len'] == 1:
        enable_max = True
    else:
        enable_max = False
    if params['summary_using_keywords'] == 1:
        enable_keywords = True
    else:
        enable_keywords = False
    if params['layer_norm'] == 1:
        layer_norm= True
    else:
        layer_norm = False
    if params['watch_rnn_output'] == 1:
        watch_rnn_output = True
    else:
        watch_rnn_output = False
        
    x_raw, y_raw, target_top_raw, target_bottom_raw, df, labels = data_helper.load_data_and_labels(dataset,params['max_length'],params['max_summary_length'],enable_max,enable_keywords)
    x_test_raw, y_test_raw, target_top_test_raw,target_bottom_test_raw, df_raw,labels_raw=data_helper.load_data_and_labels(testset,params['max_length'],params['max_summary_length'],enable_max,enable_keywords)
    word_counts = {}
    
    count_words(word_counts, x_raw)
            
    logging.info("Size of Vocabulary: {}".format(len(word_counts)))

    """Step 1: pad each sentence to the same length and map each word to an id"""
    max_document_length = max([len(x.split(' ')) for x in x_raw])
    logging.info('The maximum length of all sentences: {}'.format(max_document_length))
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
    int_top_summaries, word_count, unk_count = convert_to_ints(target_top_raw,vocab_to_int, word_count, unk_count)
    int_bottom_summaries, word_count, unk_count = convert_to_ints(target_bottom_raw,vocab_to_int, word_count, unk_count)
    int_top_test_summaries, word_count, unk_count = convert_to_ints(target_top_test_raw,vocab_to_int, word_count, unk_count)
    int_bottom_test_summaries, word_count, unk_count = convert_to_ints(target_bottom_test_raw,vocab_to_int, word_count, unk_count)
    
    int_texts, word_count, unk_count = convert_to_ints(x_raw,vocab_to_int, word_count, unk_count, eos=True)
    int_test_texts, word_count, unk_count = convert_to_ints(x_test_raw,vocab_to_int, word_count, unk_count, eos=True)
    unk_percent = round(unk_count/word_count,4)*100

    logging.info("Total number of words in texts: {}".format(word_count))
    logging.info("Total number of UNKs in  texts: {}".format(unk_count))
    logging.info("Percent of words that are UNK: {0:.2f}%".format(unk_percent))
    
    """Step 1: pad each sentence to the same length and map each word to an id"""

    x_int = pad_sentence_batch(vocab_to_int,int_texts)
    x_test_int = pad_sentence_batch(vocab_to_int,int_test_texts)
    target_top_int = pad_sentence_batch(vocab_to_int,int_top_summaries)
    target_bottom_int = pad_sentence_batch(vocab_to_int,int_bottom_summaries)
    target_top_test_int = pad_sentence_batch(vocab_to_int,int_top_test_summaries)
    target_bottom_test_int = pad_sentence_batch(vocab_to_int,int_bottom_test_summaries)
    
    x = np.array(x_int)
    x_test= np.array(x_test_int)
    y = np.array(y_raw)
    y_test = np.array(y_test_raw)
    target_top = np.array(target_top_int)
    target_bottom = np.array(target_bottom_int)
    target_top_test = np.array(target_top_test_int)
    target_bottom_test = np.array(target_bottom_test_int)
    t = np.array(list(len(x) for x in x_int))
    t_test = np.array(list(len(x) for x in x_test_int))
    s = np.array(list(params['max_summary_length'] for x in x_int))
    s_test = np.array(list(params['max_summary_length'] for x in x_test_int))

    """Step 2: shuffle the train set and split the train set into train and dev sets"""
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    target_top_shuffled = target_top[shuffle_indices]
    target_bottom_shuffled = target_bottom[shuffle_indices]
    t_shuffled = t[shuffle_indices]

    s_shuffled = s[shuffle_indices]
    x_train, x_dev, y_train, y_dev,target_top_train, target_top_dev,target_bottom_train,target_bottom_dev, t_train, t_dev, s_train, s_dev = train_test_split(x_shuffled, y_shuffled,target_top_shuffled,target_bottom_shuffled, t_shuffled,s_shuffled, test_size=0.1)

    """Step 3: save the labels into labels.json since predict.py needs it"""
    with open('./labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))
    logging.info('target_train: {}, target_dev: {}, target_test: {}'.format(len(target_top_train), len(target_top_dev), len(target_top_test)))
    logging.info('t_train: {}, t_dev: {}, t_test: {}'.format(len(t_train), len(t_dev), len(t_test)))
    logging.info('s_train: {}, s_dev: {}, s_test: {}'.format(len(s_train), len(s_dev), len(s_test)))

    """Step 4: build a graph and cnn object"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = seq2CNN(
                num_classes=y_train.shape[1],
                max_summary_length=params['max_summary_length'],
                rnn_size=params['rnn_size'],
                rnn_num_layers=params['rnn_num_layers'],
                vocab_to_int = vocab_to_int,
                num_filters=params['num_filters'],
                vocab_size=len(vocab_to_int),
                embedding_size=params['embedding_dim'],
                layer_norm=layer_norm
                )
            global_step = tf.Variable(0, name="global_step", trainable=False)            
            num_batches_per_epoch = int((len(x_train)-1)/params['batch_size']) + 1
            epsilon=params['epsilon']
            learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step,params['num_epochs']*num_batches_per_epoch, 0.95, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate,epsilon)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            cnn_gradients, cnn_variables = zip(*optimizer.compute_gradients(cnn.loss))
            seq_top_gradients, seq_top_variables = zip(*optimizer.compute_gradients(cnn.seq_top_loss))
            seq_bottom_gradients, seq_bottom_variables = zip(*optimizer.compute_gradients(cnn.seq_bottom_loss))    
            
            cnn_gradients, _ = tf.clip_by_global_norm(cnn_gradients, 7.0)
            seq_top_gradients, _ = tf.clip_by_global_norm(seq_top_gradients, 7.0)
            seq_bottom_gradients, _ = tf.clip_by_global_norm(seq_bottom_gradients, 7.0)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(zip(cnn_gradients, cnn_variables), global_step=global_step)
                seq_top_train_op = optimizer.apply_gradients(zip(seq_top_gradients, seq_top_variables), global_step=global_step)
                seq_bottom_train_op = optimizer.apply_gradients(zip(seq_bottom_gradients, seq_bottom_variables), global_step=global_step)
                
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "result_" + timestamp))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # One training step: train the model with one batch
            def train_step(x_batch, y_batch,target_top_batch,target_bottom_batch,t_batch,s_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.targets_top: target_top_batch,
                    cnn.targets_bottom: target_bottom_batch,
                    cnn.text_length: t_batch,
                    cnn.summary_length: s_batch,
                    cnn.batch_size: len(x_batch),
                    cnn.dropout_keep_prob: params['dropout_keep_prob'],
                    cnn.is_training: True}
                _, logits, step, loss,seq_top_loss,seq_bottom_loss, acc = sess.run([train_op,cnn.training_logits, global_step, cnn.loss,cnn.seq_top_loss,cnn.seq_bottom_loss, cnn.accuracy], feed_dict)
                return loss, seq_top_loss,seq_bottom_loss, acc
            def seq_top_train_step(x_batch, y_batch,target_top_batch,target_bottom_batch,t_batch,s_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.targets_top: target_top_batch,
                    cnn.targets_bottom: target_bottom_batch,
                    cnn.text_length: t_batch,
                    cnn.summary_length: s_batch,
                    cnn.batch_size: len(x_batch),
                    cnn.dropout_keep_prob: params['dropout_keep_prob'],
                    cnn.is_training: True}
                _, logits, step, loss,seq_top_loss,seq_bottom_loss, acc = sess.run([seq_top_train_op,cnn.training_logits, global_step, cnn.loss, cnn.seq_top_loss,cnn.seq_bottom_loss, cnn.accuracy], feed_dict)
                return loss, seq_top_loss,seq_bottom_loss, acc
            def seq_bottom_train_step(x_batch, y_batch,target_top_batch,target_bottom_batch,t_batch,s_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.targets_top: target_top_batch,
                    cnn.targets_bottom: target_bottom_batch,
                    cnn.text_length: t_batch,
                    cnn.summary_length: s_batch,
                    cnn.batch_size: len(x_batch),
                    cnn.dropout_keep_prob: params['dropout_keep_prob'],
                    cnn.is_training: True}
                _, logits, step, loss,seq_top_loss,seq_bottom_loss, acc = sess.run([seq_bottom_train_op,cnn.training_logits, global_step, cnn.loss, cnn.seq_top_loss,cnn.seq_bottom_loss, cnn.accuracy], feed_dict)
                return loss, seq_top_loss,seq_bottom_loss, acc

            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch, y_batch,target_top_batch,target_bottom_batch,t_batch,s_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.targets_top: target_top_batch,
                    cnn.targets_bottom: target_bottom_batch,
                    cnn.text_length: t_batch,
                    cnn.summary_length: s_batch,
                    cnn.batch_size: len(x_batch),
                    cnn.dropout_keep_prob: 1.0,
                    cnn.is_training: False}
                step, loss, seq_top_loss,seq_bottom_loss, acc, num_correct,examples = sess.run([global_step, cnn.loss, cnn.seq_top_loss,cnn.seq_bottom_loss, cnn.accuracy, cnn.num_correct,cnn.training_logits],feed_dict)
                if watch_rnn_output == True:
                    pad = vocab_to_int['PAD']
                    result =  " ".join([int_to_vocab[j] for j in examples[0] if j != pad])
                    logging.info('{}'.format(result))
                return num_correct

            # Save the word_to_id map since predict.py needs it
            vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = data_helper.batch_iter(list(zip(x_train,y_train, target_top_train,target_bottom_train,t_train, s_train)), params['batch_size'],params['num_epochs'])
            best_accuracy, best_at_step = 0, 0

            """Step 6: train the cnn model with x_train and y_train (batch by batch)"""
            for train_batch in train_batches:
                x_train_batch, y_train_batch,target_top_train_batch,target_bottom_train_batch, t_train_batch, s_train_batch = zip(*train_batch)
                train_loss, train_seq_top_loss,train_seq_bottom_loss, train_acc = seq_top_train_step(x_train_batch,y_train_batch,target_top_train_batch,target_bottom_train_batch, t_train_batch, s_train_batch)
                train_loss,train_seq_top_loss,train_seq_bottom_loss,train_acc = seq_bottom_train_step(x_train_batch,y_train_batch,target_top_train_batch,target_bottom_train_batch, t_train_batch, s_train_batch)
                train_loss,train_seq_top_loss,train_seq_bottom_loss, train_acc = train_step(x_train_batch,y_train_batch,target_top_train_batch,target_bottom_train_batch, t_train_batch, s_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step%params['evaluate_every'] ==0:
                    logging.critical('step: {} accuracy: {} cnn_loss: {} seq_top_loss: {} seq_bottom_loss: {}'.format(current_step, train_acc, train_loss, train_seq_top_loss,train_seq_bottom_loss))

                
                """Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
                    dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev,target_top_dev, target_bottom_dev,t_dev,s_dev)), params['batch_size'], 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch,y_dev_batch,target_top_dev_batch,target_bottom_dev_batch, t_dev_batch,s_dev_batch = zip(*dev_batch)
                        num_dev_correct = dev_step(x_dev_batch,y_dev_batch,target_top_dev_batch,target_bottom_dev_batch, t_dev_batch,s_dev_batch)
                        total_dev_correct += num_dev_correct

                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

                    """Step 6.2: save the model if it is the best based on accuracy on dev set"""
                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model at {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy is {} at step {}'.format(best_accuracy, best_at_step))

            """Step 7: predict x_test (batch by batch)"""
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test,target_top_test, target_bottom_test,t_test,s_test)), params['batch_size'], 1)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch,y_test_batch,target_top_test_batch,target_bottom_test_batch, t_test_batch,s_test_batch = zip(*test_batch)
                num_test_correct = dev_step(x_test_batch,y_test_batch,target_top_test_batch,target_bottom_test_batch, t_test_batch,s_test_batch)
                total_test_correct += num_test_correct
            path = saver.save(sess, checkpoint_prefix)
            test_accuracy = float(total_test_correct) / len(y_test)
            logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
            logging.critical('The training is complete')


if __name__ == '__main__':
    # python3 train.py ./data/consumer_complaints.csv.zip ./parameters.json
    train_cnn(sys.argv[1])
