import os
import sys
import json
import time
import logging
import data_helper
import numpy as np
import tensorflow as tf
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)  

def train_cnn(dataset_name):
    """Step 0: load sentences, labels, and training parameters"""
    dataset = '../dataset/'+dataset_name+'_csv/train.csv'
    testset = '../dataset/'+dataset_name+'_csv/test.csv'
    parameter_file = "./parameters.json"
    params = json.loads(open(parameter_file).read())
    x_raw, y_raw, df, labels = data_helper.load_data_and_labels(dataset,params['max_length'])
    x_test, y_test, df, labels = data_helper.load_data_and_labels(testset,params['max_length'])

    """Step 1: pad each sentence to the same length and map each word to an id"""
    max_document_length = max([len(x.split(' ')) for x in x_raw])
    logging.info('The maximum length of all sentences: {}'.format(max_document_length))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_raw)))
    y = np.array(y_raw)
    x_test = np.array(list(vocab_processor.fit_transform(x_test)))
    y_test = np.array(y_test)

    """Step 3: shuffle the train set and split the train set into train and dev sets"""
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)

    """Step 4: save the labels into labels.json since predict.py needs it"""
    with open('./labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    """Step 5: build a graph and cnn object"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=params['embedding_dim'],
                filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                num_filters=params['num_filters'],
                l2_reg_lambda=params['l2_reg_lambda'])

            global_step = tf.Variable(0, name="global_step", trainable=False)
            num_batches_per_epoch = int((len(x_train)-1)/params['batch_size']) + 1
            learning_rate = tf.train.exponential_decay(0.0001, global_step,params['num_epochs']*num_batches_per_epoch, 0.95, staircase=True)
            #learning_rate = params['learning_rate']
            optimizer = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            timestamp = str(int(time.time()))
            #timestamp = 'H'
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # One training step: train the model with one batch
            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.is_training: True,
                    cnn.dropout_keep_prob: params['dropout_keep_prob']}
                _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
                return acc,loss

            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch, y_batch):
                feed_dict = {cnn.input_x: x_batch, 
                             cnn.input_y: y_batch,
                             cnn.is_training: False,
                             cnn.dropout_keep_prob: 1.0}
                step, loss, acc, num_correct = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct],
                                                        feed_dict)
                return num_correct

            # Save the word_to_id map since predict.py needs it
            vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'],
                                                   params['num_epochs'])
            best_accuracy, best_at_step = 0, 0


            """Step 6: train the cnn model with x_train and y_train (batch by batch)"""
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                train_acc,  train_loss = train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)
                """Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
                    logging.critical('step: {} accuracy: {} cnn_loss: {} '.format(current_step, train_acc, train_loss))
                    dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        num_dev_correct = dev_step(x_dev_batch, y_dev_batch)
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
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                num_test_correct = dev_step(x_test_batch, y_test_batch)
                total_test_correct += num_test_correct
            #path = saver.save(sess, checkpoint_prefix)
            test_accuracy = float(total_test_correct) / len(y_test)
            logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
            logging.critical('The training is complete')


if __name__ == '__main__':
    # python3 train.py ./data/consumer_complaints.csv.zip ./parameters.json
    train_cnn(sys.argv[1])
