import os
import sys
import json
import time
import logging
import data_helper
import numpy as np
import tensorflow as tf
from text_cnn import seq2CNN
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
    dataset = './dataset/'+dataset_name+'_csv/train.csv'
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
        
    x_raw, y_raw, target_raw, df, labels = data_helper.load_data_and_labels(dataset,params['max_length'],params['max_summary_length'],enable_max,enable_keywords)
    word_counts = {}
    
    count_words(word_counts, x_raw)
            
    logging.info("Size of Vocabulary: {}".format(len(word_counts)))

    # Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better 
    # (https://github.com/commonsense/conceptnet-numberbatch)
    #embeddings_index = {}
    #with open('./dataset/embeddings/numberbatch-en.txt', encoding='utf-8') as f:
        #for line in f:
            #values = line.split(' ')
            #word = values[0]
            #embedding = np.asarray(values[1:], dtype='float32')
            #embeddings_index[word] = embedding
    #max_document_length = max([len(x.split(' ')) for x in x_raw])
    
    # Find the number of words that are missing from CN, and are used more than our threshold.
    #missing_words = 0
    #threshold = params['missing_threshhold']

    #for word, count in word_counts.items():
       # if count > threshold:
            #if word not in embeddings_index:
                #missing_words += 1
            
    #missing_ratio = round(missing_words/len(word_counts),4)*100
            
   # logging.info("Number of words missing from CN: {}".format(missing_words))
    #logging.info("Percent of words that are missing from vocabulary: {0:.2f}%".format(missing_ratio))

    #dictionary to convert words to integers
    """Step 1: pad each sentence to the same length and map each word to an id"""
    max_document_length = max([len(x.split(' ')) for x in x_raw])
    logging.info('The maximum length of all sentences: {}'.format(max_document_length))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor.fit_transform(x_raw)
    vocab_to_int = vocab_processor.vocabulary_._mapping
    #print(vocab_to_int)
    #value = 0
    #for word, count in word_counts.items():
        #if count >= threshold:
            #vocab_to_int[word] = value
            #value += 1
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
    
    # Need to use 300 for embedding dimensions to match CN's vectors.
    #embedding_dim = 300
    #nb_words = len(vocab_to_int)

    # Create matrix with default values of zero
    #word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
    #for word, i in vocab_to_int.items():
        #if word in embeddings_index:
            #word_embedding_matrix[i] = embeddings_index[word]
        #else:
            # If word not in CN, create a random embedding for it
           # new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
           # embeddings_index[word] = new_embedding
            #word_embedding_matrix[i] = new_embedding

    # Check if value matches len(vocab_to_int)
    #logging.info(len(word_embedding_matrix))

    # Apply convert_to_ints to clean_summaries and clean_texts
    word_count = 0
    unk_count = 0
    int_summaries, word_count, unk_count = convert_to_ints(target_raw,vocab_to_int, word_count, unk_count)
    int_texts, word_count, unk_count = convert_to_ints(x_raw,vocab_to_int, word_count, unk_count, eos=True)
    unk_percent = round(unk_count/word_count,4)*100

    logging.info("Total number of words in texts: {}".format(word_count))
    logging.info("Total number of UNKs in  texts: {}".format(unk_count))
    logging.info("Percent of words that are UNK: {0:.2f}%".format(unk_percent))
    
    """Step 1: pad each sentence to the same length and map each word to an id"""

    x_int = pad_sentence_batch(vocab_to_int,int_texts)
    target_int = pad_sentence_batch(vocab_to_int,int_summaries)
    x = np.array(x_int)
    y = np.array(y_raw)
    target = np.array(target_int)
    t = np.array(list(len(x) for x in x_int))
    s = np.array(list(params['max_summary_length'] for x in x_int))


    
    """Step 2: split the original dataset into train and test sets"""
    x_, x_test, y_, y_test,target_, target_test, t_,t_test,s_,s_test = train_test_split(x, y,target, t, s, test_size=0.1, random_state=42)

    """Step 3: shuffle the train set and split the train set into train and dev sets"""
    shuffle_indices = np.random.permutation(np.arange(len(y_)))
    x_shuffled = x_[shuffle_indices]
    y_shuffled = y_[shuffle_indices]
    target_shuffled = target_[shuffle_indices]
    t_shuffled = t_[shuffle_indices]
    s_shuffled = s_[shuffle_indices]
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
                max_summary_length=params['max_summary_length'],
                rnn_size=params['rnn_size'],
                rnn_num_layers=params['rnn_num_layers'],
                vocab_to_int = vocab_to_int,
                num_filters=params['num_filters'],
                vocab_size=len(word_counts),
                embedding_size=params['embedding_dim'])

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # One training step: train the model with one batch
            def train_step(x_batch, y_batch,target_batch,t_batch,s_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.targets: target_batch,
                    cnn.text_length: t_batch,
                    cnn.summary_length: s_batch,
                    cnn.batch_size: len(x_batch),
                    cnn.dropout_keep_prob: params['dropout_keep_prob']}
                _, logits, step, loss, acc = sess.run([train_op,cnn.training_logits, global_step, cnn.loss, cnn.accuracy], feed_dict)
                return loss, acc

            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch, y_batch,target_batch, t_batch,s_batch):
                feed_dict = {
                    cnn.input_x: x_batch, 
                    cnn.input_y: y_batch,
                    cnn.targets: target_batch,
                    cnn.text_length: t_batch,
                    cnn.summary_length: s_batch,
                    cnn.batch_size: len(x_batch),
                    cnn.dropout_keep_prob: 1.0}
                step, loss, acc, num_correct = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct],
                                                        feed_dict)
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
                train_loss, train_acc = train_step(x_train_batch, y_train_batch,target_train_batch,t_train_batch,s_train_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step%100==0:
                    logging.critical('step: {} accuracy: {} loss: {}'.format(current_step, train_acc, train_loss))

                
                """Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
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

            """Step 7: predict x_test (batch by batch)"""
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test,target_test,t_test,s_test)), params['batch_size'], 1)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch,target_test_batch, t_test_batch,s_test_batch = zip(*test_batch)
                num_test_correct = dev_step(x_test_batch, y_test_batch,target_test_batch,t_test_batch,s_test_batch)
                total_test_correct += num_test_correct
            path = saver.save(sess, checkpoint_prefix)
            test_accuracy = float(total_test_correct) / len(y_test)
            logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
            logging.critical('The training is complete')


if __name__ == '__main__':
    # python3 train.py ./data/consumer_complaints.csv.zip ./parameters.json
    train_cnn(sys.argv[1])
