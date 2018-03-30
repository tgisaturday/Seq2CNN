import re
import time
import logging
import numpy as np
import pandas as pd
import random
import json
import sys
import os
from collections import Counter
from konlpy.tag import Kkma
from konlpy.tag import Mecab
from konlpy.tag import Komoran
from konlpy.utils import pprint


def clean_str(s,max_length):
    """Clean sentence"""
    global counter_konlpy
    global total_dataset
    global stopwords
    s = re.sub('[0-9]', '', s)
    #s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    #s = re.sub(r"\'s", " \'s", s)
    #s = re.sub(r"\'ve", " \'ve", s)
    #s = re.sub(r"n\'t", " n\'t", s)
    #s = re.sub(r"\'re", " \'re", s)
    #s = re.sub(r"\'d", " \'d", s)
    #s = re.sub(r"\'ll", " \'ll", s)
    #s = re.sub(r",", " , ", s)
    #s = re.sub(r"!", " ! ", s)
    #s = re.sub(r"\(", " \( ", s)
    #s = re.sub(r"\)", " \) ", s)
    #s = re.sub(r"\?", " \? ", s)
    #s = re.sub(r"\s{2,}", " ", s)
    #s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
    #s = re.sub(r'[^\x00-\x7F]+', "", s)
    s = s.strip()
    #s = s.strip().lower()

    result = []
    #result = s.split(' ')
    mecab = Mecab()
    result = mecab.nouns(s)   
    
    if len(result) >= max_length:
        result = result[0:max_length]
    elif len(result) < max_length:
        result = result + ["<PAD/>"] * (max_length - len(result) -2)
    counter_konlpy += 1
    sys.stdout.write("\rParsed: %d / %d" %(counter_konlpy, total_dataset))
    sys.stdout.flush()
    
    return ' '.join(result)



def load_data_and_labels(foldername,max_length):
    """Load sentences and labels"""
    columns = ['section', 'class', 'subclass', 'abstract']
    selected = ['section', 'abstract']
    global counter_konlpy
    global total_dataset
    #global stopwords
    #stopword_file = "./stopwords.json"
    #stopwords = tuple(json.loads(open(stopword_file).read()))
    file_list = []
    for path, dirs, files in os.walk(foldername):
        if files:
            for filename in files:
                fullname = os.path.join(path, filename)
                file_list.append(fullname)  
               
    random.shuffle(file_list)


    data = []
    print("Listing all datas in dataset.")
    start = time.time()
    for filename in file_list:
        fp = open(filename, 'r', encoding='utf-8')
        temp = fp.readlines()
        data.append([filename.split('/')[3], filename.split('/')[3]+filename.split('/')[4],filename.split('/')[3]+filename.split('/')[4]+ filename.split('/')[5], ''.join(temp)])
        fp.close()

    df = pd.DataFrame(data, columns=columns)
    print("Execution time = {0:.5f}".format(time.time() - start))

    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)  # Drop non selected columns
    df = df.dropna(axis=0, how='any', subset=selected)  # Drop null rows
    df = df.reindex(np.random.permutation(df.index))  # Shuffle the dataframe

    # Map the actual labels to one hot labels
    labels = sorted(list(set(df[selected[0]].tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    print("Preprocessing Dataset.")
    start = time.time()
    counter_konlpy = 0
    total_dataset = len(file_list)
    x_raw = df[selected[1]].apply(lambda x: clean_str(x,max_length)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    print("\nExecution time = {0:.5f}".format(time.time() - start))
    return x_raw, y_raw, df, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
            
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    dataset = './dataset/abstract'
    #dataset = './dataset/description'
    load_data_and_labels(dataset)
