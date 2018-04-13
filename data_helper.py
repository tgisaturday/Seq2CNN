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
from summa.summarizer import summarize
from summa.keywords import keywords
def clean_str(text,max_length,enable_max):
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
    
    if enable_max :
        if len(text) >= max_length:
            text = text[0:max_length]
        elif len(text) < max_length:
            text = text + ["PAD"] * (max_length - len(text))
            text = text[0:max_length]
        
    return ' '.join(text).strip()
def gen_keywords(text,max_length):
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
    text = keywords(text,split = True)
    text = " ".join(text)
    text = text.split()
    if len(text) >= max_length:
        text = text[0:max_length]
    elif len(text) < max_length:
        text = text + ["PAD"] * (max_length - len(text))
        text = text[0:max_length]

    return '<GO> '+' '.join(text).strip()

def gen_summary(text,max_length):
    """Clean sentence"""
    sentence = summarize(text)
    bow = sentence
    bow = bow.lower()
    bow = bow.split()
    bow = bow + keywords(text,split = True)
    bow = bow + text.lower().split()
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
    text = ['GO']+text
    if len(text) >= max_length:
        text = text[0:max_length]
    else:
        text = text + ["PAD"] * (max_length - len(text))
        text = text[0:max_length]
    return ' '.join(text)

def load_data_and_labels(filename,max_length,max_summary_length,enable_max,enable_keywords):
    """Load sentences and labels"""
    df = pd.read_csv(filename, names=['label', 'company', 'text'], dtype={'text': object})
    selected = ['label', 'company','text']
    non_selected = list(set(df.columns) - set(selected))
    df = df.drop(non_selected, axis=1) # Drop non selected columns
    df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
    df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe
    # Map the actual labels to one hot labels
    labels = sorted(list(set(df[selected[0]].tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    x_raw = df[selected[2]].apply(lambda x: clean_str(x,max_length,enable_max)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()


    if enable_keywords:
        target_raw = df[selected[2]].apply(lambda x: gen_summary(x,max_summary_length)).tolist()
    else:
        target_raw = df[selected[1]].apply(lambda x: clean_str(x,max_summary_length,True)).tolist()

    return x_raw, y_raw,target_raw, df, labels

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
