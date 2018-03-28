import re
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import data_helper
import random
import time
from tensorflow.contrib import learn
from konlpy.tag import Kkma
from konlpy.tag import Mecab
from konlpy.utils import pprint

logging.getLogger().setLevel(logging.INFO)

def clean_str(s):
    """Clean sentence"""
    global counter_konlpy
    global total_dataset
    s = re.sub('[0-9]', '', s)
    #kkma = Kkma()
    # komoran  = Komoran()
    # twitter = Twitter()
    mecab = Mecab()
    #print(' '.join(kkma.nouns(s)))
    result = []
    result = mecab.nouns(s)
    if len(result) > 1000:
        result = result[0:1000]
    elif len(result) < 1000:
        result = result + ["<PAD/>"] * (1000 - len(result)-1)    
    counter_konlpy += 1
    sys.stdout.write("\r Parsed: %d / %d" %(counter_konlpy, total_dataset))
    sys.stdout.flush()
    return ' '.join(result)

def predict_unseen_data():
    """Step 0: load trained model and parameters"""
    params = json.loads(open('./parameters.json').read())
    checkpoint_dir = sys.argv[1]
    if not checkpoint_dir.endswith('/'):
        checkpoint_dir += '/'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
    logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

    """Step 1: load data for prediction"""
    test_file = "./dataset/description/"
    columns = ['section', 'class', 'subclass', 'description']
    selected = ['section', 'description']
    test_list = []
    for path, dirs, files in os.walk(test_file):
        if files:
            for filename in files:
                fullname = os.path.join(path,filename)
                if"2017" in filename.split('-')[0]:
                    test_list.append(fullname)
    random.shuffle(test_list)
    #test_list = test_list[0:10000]
    data = []
    print("Listing all datas in testset.")
    start = time.time()
    for filename in test_list:
        fp = open(filename,'r',encoding='utf-8')
        temp = fp.readlines()
        data.append([filename.split('/')[3], filename.split('/')[4], filename.split('/')[5], ' '.join(temp)])
        fp.close()
    df = pd.DataFrame(data, columns=columns)
    print("Execution time = {0:.5f}".format(time.time() - start))

    # labels.json was saved during training, and it has to be loaded during prediction
    labels = json.loads(open('./labels.json').read())
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    
    global counter_konlpy
    global total_dataset
    start = time.time()
    counter_konlpy = 0
    total_dataset = len(test_list)
    #x_raw = [example['abstract'] for example in test_examples]
    #x_test = [data_helper.clean_str(x) for x in x_raw]
    x_test = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y_test = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    print("\nExecution time = {0:.5f}".format(time.time() - start))
    
    logging.info('The number of x_test: {}'.format(len(x_test)))
    logging.info('The number of y_test: {}'.format(len(y_test)))

    vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_test)))

    """Step 2: compute the predictions"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    if y_test is not None:
        y_test = np.argmax(y_test, axis=1)
        correct_predictions = sum(all_predictions == y_test)
        logging.critical('The accuracy is: {}'.format(correct_predictions / float(len(y_test))))
        logging.critical('The prediction is complete')


if __name__ == '__main__':
    # python3 predict.py ./trained_model_1478649295/ ./data/small_samples.json
    predict_unseen_data()
