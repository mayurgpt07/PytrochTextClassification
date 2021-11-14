import io
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from TextProcessing.DataAndProcessing import text_cleaner
import numpy as np
import pandas as pd
from gensim.models import FastText
from gensim.utils import tokenize

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from time import time
from sklearn import metrics

import config
import dataset
import engine
import lstm

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        # print(np.array(list(map(float, tokens[1:]))))
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    # print(y_pred_tag, y_test)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def feature_eng(df):
    df.drop(columns = ['id', 'location'], inplace = True)
    df['keyword'].fillna('No Keyword', inplace = True)
    df['text'].fillna('No Text', inplace = True)

    df = text_cleaner(df, column_name='text', 
                            remove_stopwords=True, 
                            append_stopwords=True,
                            listOfStopWords = ['#'], 
                            remove_digits=True, 
                            do_lemmatization=True)

    df = text_cleaner(df,column_name='keyword', 
                            remove_stopwords=True, 
                            append_stopwords=True,
                            listOfStopWords = ['#'], 
                            remove_digits=True, 
                            do_lemmatization=True)

    df['full_text'] = df.apply(lambda x: x['text_processed'] + ' ' + x['keyword_processed'] if x['keyword_processed'] != 'keyword' else x['text_processed'], axis = 1)
    df.drop(columns = ['keyword', 'text','text_processed', 'keyword_processed'], inplace = True)

    df = text_cleaner(df,column_name='full_text', 
                            remove_stopwords=True, 
                            append_stopwords=True,
                            listOfStopWords = ['#'], 
                            remove_digits=False, 
                            do_lemmatization=True)

    df.drop(columns = ['full_text'], inplace = True)
    return df

def create_embedding_matrix(word_index, embedding_model):
    embedding_matrix = np.zeros((len(word_index)+1, 300))
    for word, i in word_index.items():
        # print(word, i)
        embedding_dict = list(embedding_model.wv.vocab.keys())
        if word in embedding_dict:
            embedding_matrix[i] = embedding_model.wv[word]
    return embedding_matrix

def run(df, fold):
    #1856, 5696
    train_df = df[df.kfold != fold].reset_index(drop = True)
    print(len(train_df))
    train_df = train_df.loc[:5695, :]
    valid_df = df[df.kfold == fold].reset_index(drop = True)
    valid_df = valid_df.loc[:1471, :]
    print(len(train_df), len(valid_df))
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.full_text_processed.values.tolist())

    sentence_list = df['full_text_processed'].to_list()

    sentence_word_list = []

    for sentence in sentence_list:
        sentence_word_list.append(list(tokenize(sentence, lowercase = True)))

    ## Calculate the max legth sentence
    final_list = list(map(lambda x: len(x), sentence_list))
    # print('Max len sentence', max(final_list))

    maxLength = int(max(final_list))
    print(maxLength)

    xtrain = tokenizer.texts_to_sequences(train_df.full_text_processed.values)
    xtest = tokenizer.texts_to_sequences(valid_df.full_text_processed.values)

    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen = maxLength)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen = maxLength)

    train_dataset = dataset.IMDBDataset(reviews=xtrain, targets=train_df.target.values)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config.TRAIN_BATCH_SIZE, num_workers = 2)

    valid_dataset = dataset.IMDBDataset(reviews=xtest, targets=valid_df.target.values)

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = config.VALID_BATCH_SIZE, num_workers = 2)

    print('Loading Embeddings')
    # embedding_dict = load_vectors('./crawl-300d-2M.vec')
    embedding_model = FastText(min_count=1, size = 300)
    embedding_model.build_vocab(sentence_word_list, progress_per=50)
    embedding_model.train(sentence_word_list, total_examples=len(sentence_word_list), epochs=30, report_delay=1)
    print('Embeddings Loaded')
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_model)
    print('Shape of embedding', embedding_matrix.shape)
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = lstm.LSTM(embedding_matrix, config.TRAIN_BATCH_SIZE, maxLength, config.HIDDEN_SIZE, config.LAYERS)
    print(device)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 3)

    print('Training model')

    best_accuracy = 0

    early_stopping_counter = 0

    for epoch in range(config.EPOCHS):
        t1 = time()
        engine.train(train_data_loader, model, optimizer, device, config.TRAIN_BATCH_SIZE)
        
        curr_lr = optimizer.param_groups[0]['lr']
        print('Current Learning Rate: ', curr_lr)
        
        outputs, targets, valid_loss = engine.evaluate(valid_data_loader, model, device, config.VALID_BATCH_SIZE, optimizer, scheduler)
        
        scheduler.step(valid_loss/len(valid_data_loader))
        
        flat_outputs = [item for sublist in outputs for item in sublist]
        prob = np.array(flat_outputs)
        # print(prob)
        outputs = np.array(flat_outputs) >= 0.5
        # print(targets[:5], outputs[:5])
        accuracy = metrics.accuracy_score(targets, outputs)
        # print(torch.FloatTensor(flat_outputs), torch.FloatTensor(targets))
        acc = binary_acc(torch.FloatTensor(flat_outputs), torch.FloatTensor(targets))
        auc = metrics.roc_auc_score(targets, prob)

        print('Fold: ', fold, ' EPOCH: ', epoch, ' Accuracy Score: ', accuracy, acc, ' AUC:', auc)
        t2 = time()

        print('Time Taken', t2-t1)
        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        # else:
        #     early_stopping_counter +=1

        # if early_stopping_counter > 2:
        #     print(early_stopping_counter)
        #     break

if __name__ == '__main__':

    df = pd.read_csv('./train_folds.csv')
    df = feature_eng(df)
    run(df, fold=0)
    run(df, fold=1)
    run(df, fold=2)
    run(df, fold=3)