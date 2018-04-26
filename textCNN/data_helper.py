import re
import time
import logging
import numpy as np
import pandas as pd
import random
import json
import sys
import os
from nltk.corpus import stopwords
from collections import Counter
from contractions import get_contractions

def empty_remover(text):
    removed = []
    for word in text:
        if word != '':
            removed.append(word)
    return removed

def clean_str(text):
    """Clean sentence"""
    text = text.lower()
    text = text.split()
    new_text = []
    contractions = get_contractions()
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = " ".join(new_text)
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    text = text.split(' ')
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops] 
    
    text = empty_remover(text)

    return ' '.join(text).strip()

def load_data_and_labels(filename, dataset_name):
    """Load sentences and labels"""
    if dataset_name == 'ag_news' or dataset_name == 'dbpedia' or dataset_name == 'sogou_news' or dataset_name == 'amazon_review_full' or dataset_name == 'amazon_review_polarity' :
        df = pd.read_csv(filename, names=['label', 'title', 'text'], dtype={'title': object,'text': object})
        selected = ['label', 'title','text']

        non_selected = list(set(df.columns) - set(selected))
        df = df.drop(non_selected, axis=1) # Drop non selected columns
        df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
        df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe
        # Map the actual labels to one hot labels
        labels = sorted(list(set(df[selected[0]].tolist())))
        one_hot = np.zeros((len(labels), len(labels)), int)
        np.fill_diagonal(one_hot, 1)
        label_dict = dict(zip(labels, one_hot))

        x_raw = df[selected[2]].apply(lambda x: clean_str(x)).tolist()
        y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
            
    elif dataset_name == 'yelp_review_full' or dataset_name == 'yelp_review_polarity':
        df = pd.read_csv(filename, names=['label','text'], dtype={'text': object})
        selected = ['label','text']
        non_selected = list(set(df.columns) - set(selected))
        df = df.drop(non_selected, axis=1) # Drop non selected columns
        df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
        df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe
        # Map the actual labels to one hot labels
        labels = sorted(list(set(df[selected[0]].tolist())))
        one_hot = np.zeros((len(labels), len(labels)), int)
        np.fill_diagonal(one_hot, 1)
        label_dict = dict(zip(labels, one_hot))

        x_raw = df['text'].apply(lambda x: clean_str(x)).tolist()
        y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

    elif dataset_name == 'yahoo_answers':
        df = pd.read_csv(filename, names=['label', 'title', 'content','answer'], dtype={'title': object,'answer': object,'content': object})
        selected = ['label', 'title','content','answer']
        
        non_selected = list(set(df.columns) - set(selected))
        df['temp'] = df[['content','answer']].apply(lambda x: ' '.join(str(v) for v in x), axis=1)
        df['merged'] = df[['temp','title']].apply(lambda x: ' '.join(str(v) for v in x), axis=1)
        df = df.drop(non_selected, axis=1) # Drop non selected columns
        df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
        df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe

        labels = sorted(list(set(df[selected[0]].tolist())))
        one_hot = np.zeros((len(labels), len(labels)), int)
        np.fill_diagonal(one_hot, 1)
        label_dict = dict(zip(labels, one_hot))

        x_raw = df['temp'].apply(lambda x: clean_str(x)).tolist()
        y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

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


