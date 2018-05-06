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
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

def empty_remover(text):
    removed = []
    for word in text:
        if word != '':
            removed.append(word)
    return removed

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
    text = re.sub(r'[0-9]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    text = text.split(' ')
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops] 
    
    text = empty_remover(text)
    if enable_max :
        if len(text) >= max_length:
            text = text[0:max_length]
        elif len(text) < max_length:
            text = text + ["PAD"] * (max_length - len(text))
            text = text[0:max_length]
        
    return ' '.join(text).strip()

def gen_summary(text,max_length):
    """Clean sentence"""
    try:
        sentence = summarize(text, word_count=max_length)
    except:
        sentence = text

    bow = sentence
    bow = bow.lower()
    bow = bow.split()
    #bow = bow + keywords(text,split = True)
    bow = bow + text.lower().split()
    new_text = []
    contractions = get_contractions()
    for word in bow:
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
    text = re.sub(r'[0-9]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = text.split(' ')
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = ['GO']+text
    text = empty_remover(text)
    if len(text) >= max_length:
        text = text[0:max_length]
    else:
        text = text + ["PAD"] * (max_length - len(text))
        text = text[0:max_length]
    return ' '.join(text)

def remove_short(text,min_length):
    total = len(text.split(' '))
    if total < min_length:
        return 'N/A'
    else:
        return total
    
def shrink_df(label,label_count,data_per_class):
    count = label_count.get(label)
    if count == None:
        label_count[label]=1
        return label
    elif count < data_per_class:
        label_count[label]+=1
        return label
    else:
        return 'N/A'
    
def load_data_and_labels(filename,dataset_name,max_length,max_summary_length,enable_max):
    """Load sentences and labels"""
    label_count={}
    parameter_file = "./parameters.json"
    params = json.loads(open(parameter_file).read())
    if dataset_name == 'ag_news' or dataset_name == 'dbpedia' or dataset_name == 'sogou_news' or dataset_name == 'amazon_review_full' or dataset_name == 'amazon_review_polarity' :
        df = pd.read_csv(filename, names=['label', 'title', 'text'], dtype={'title': object,'text': object})
        selected = ['label', 'title','text','too_short','to_drop']
        non_selected = list(set(df.columns) - set(selected))
        df = df.drop(non_selected, axis=1) # Drop non selected columns        
        df['too_short']= df[selected[2]].apply(lambda x: (remove_short(x,max_summary_length)))
        df['too_short']=df['too_short'].replace('N/A',np.NaN)
        df = df.dropna(axis=0, how='any') # Drop null rows        
        df['to_drop']= df[selected[0]].apply(lambda y: (shrink_df(y,label_count,params['data_per_class'])))
        df['to_drop']=df['to_drop'].replace('N/A',np.NaN)
        df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
        df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe
        for key,value in label_count.items():
            print("{} : {}".format(key,value))
    # Map the actual labels to one hot labels
        labels = sorted(list(set(df[selected[0]].tolist())))
        one_hot = np.zeros((len(labels), len(labels)), int)
        np.fill_diagonal(one_hot, 1)
        label_dict = dict(zip(labels, one_hot))

        x_raw = df[selected[2]].apply(lambda x: clean_str(x,max_length,enable_max)).tolist()
        y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
        start = time.time()
        target_raw = df[selected[2]].apply(lambda x: gen_summary(x,max_summary_length)).tolist()
        print("\nExecution time for summary generation = {0:.5f}".format(time.time() - start))
            
    elif dataset_name == 'yelp_review_full' or dataset_name == 'yelp_review_polarity':
        df = pd.read_csv(filename, names=['label','text'], dtype={'text': object})
        selected = ['label','text','too_short','to_drop']
        non_selected = list(set(df.columns) - set(selected))
        df = df.drop(non_selected, axis=1) # Drop non selected columns        
        df['too_short']= df[selected[1]].apply(lambda x: (remove_short(x,max_summary_length)))
        df['too_short']=df['too_short'].replace('N/A',np.NaN)
        df = df.dropna(axis=0, how='any') # Drop null rows        
        df['to_drop']= df[selected[0]].apply(lambda y: (shrink_df(y,label_count,params['data_per_class'])))
        df['to_drop']=df['to_drop'].replace('N/A',np.NaN)
        df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
        df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe
        for key,value in label_count.items():
            print("{} : {}".format(key,value))
    # Map the actual labels to one hot labels
        labels = sorted(list(set(df[selected[0]].tolist())))
        one_hot = np.zeros((len(labels), len(labels)), int)
        np.fill_diagonal(one_hot, 1)
        label_dict = dict(zip(labels, one_hot))

        x_raw = df['text'].apply(lambda x: clean_str(x,max_length,enable_max)).tolist()
        y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
        start = time.time()
        target_raw = df['text'].apply(lambda x: gen_summary(x,max_summary_length)).tolist()
        print("\nExecution time for summary generation = {0:.5f}".format(time.time() - start))
            
    elif dataset_name == 'yahoo_answers':
        df = pd.read_csv(filename, names=['label', 'title', 'content','answer'], dtype={'title': object,'answer': object,'content': object})
        selected = ['label', 'title','content','answer','too_short','to_drop']
        
        non_selected = list(set(df.columns) - set(selected))
        df = df.drop(non_selected, axis=1) # Drop non selected columns        
        df['temp'] = df[['content','answer']].apply(lambda x: ' '.join(str(v) for v in x), axis=1)
        df['too_short']= df['temp'].apply(lambda x: (remove_short(x,max_summary_length)))
        df['too_short']=df['too_short'].replace('N/A',np.NaN)
        df = df.dropna(axis=0, how='any') # Drop null rows        
        df['to_drop']= df[selected[0]].apply(lambda y: (shrink_df(y,label_count,params['data_per_class'])))
        df['to_drop']=df['to_drop'].replace('N/A',np.NaN)

        df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
        df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe
        for key,value in label_count.items():
            print("{} : {}".format(key,value))
        labels = sorted(list(set(df[selected[0]].tolist())))
        one_hot = np.zeros((len(labels), len(labels)), int)
        np.fill_diagonal(one_hot, 1)
        label_dict = dict(zip(labels, one_hot))

        x_raw = df['temp'].apply(lambda x: clean_str(x,max_length,enable_max)).tolist()
        y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
        start = time.time()
        target_raw = df['temp'].apply(lambda x: gen_summary(x,max_summary_length)).tolist()
        print("\nExecution time for summary generation = {0:.5f}".format(time.time() - start))
                   
            
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
