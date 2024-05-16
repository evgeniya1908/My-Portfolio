# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:22:16 2021

@author: User
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, GRU, Conv1D, MaxPooling1D, Input, concatenate
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.saving import load_model
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import keras
import time 
import pickle


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_positives = tf.cast(possible_positives , tf.float64)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    predicted_positives = tf.cast( predicted_positives, tf.float64)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



class Learning():
    
    def __init__(self):
        super().__init__()
        self.maxWordsCount = 1000
        self.max_text_len =50
        self.tokenizer = Tokenizer(num_words=self.maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True, split=' ', char_level=False)
    

    def normalize(self, texts):
        self.tokenizer.fit_on_texts(texts)
        data = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(data, maxlen=self.max_text_len)

        
    def creat_x_y(self, fnames):
        x = []
        y = []
        k = 0
        l = 0
        for fn in fnames:
            xl = pd.ExcelFile(fn)
            sh = xl.sheet_names 
            for el in sh:
                df = xl.parse(el)
                n = df.shape[0]
                k = k+n
                l = df.shape[1]
                for i in range(n):
                    x.append(df.values[i, 0])
                    for j in range(1,l):
                        y.append(df.values[i, j])
        y = np.array(y)
        Y = y.reshape(k, l-1)
        X = self.normalize(x)
        return X, Y
    
    def creat_model(self):
        inp = Input(shape=(self.max_text_len,))
        #x = Embedding(self.maxWordsCount, 128, input_length = self.max_text_len)(inp)
        layer = Embedding(self.maxWordsCount, 128)(inp)
        layer1 = Conv1D(100, 3, activation='relu')(layer)
        layer1 = MaxPooling1D()(layer1)
        layer2 = Conv1D(100, 4, activation='relu')(layer)
        layer2 = MaxPooling1D()(layer2)
        layer3 = Conv1D(100, 5, activation='relu')(layer)
        layer3 = MaxPooling1D()(layer3)
        layer = concatenate([layer1, layer2, layer3], axis=1)
        layer = Bidirectional(GRU(64))(layer)
        out = Dense(4, activation='sigmoid')(layer)
        self.model = Model(inp, out)
        self.model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    def learn_model(self, X, Y, test):
        indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
        X = X[indeces]
        Y = Y[indeces]
        time_callback = TimeHistory()
        history = self.model.fit(X, Y, batch_size=10, epochs=4, callbacks=[time_callback], validation_data=test)
        plt.plot(history.history['accuracy'],label='Частина вiрних вiдповiдей')
        plt.plot(history.history['loss'], label='Частина втрат')
        plt.xlabel('Епоха навчання')
        plt.ylabel('Частини')
        plt.legend()
        plt.show()
        return history
    
    
    
    def test_model(self, X, Y):
        self.k = [2, 2]
        self.colum = ['телефон', 'ноутбук', 'хороший', 'плохой']
        
        #self.model = load_model('convBiGRU.h5')
        res = self.model.predict(X)
        #print(res)
        n = len(res)
        m = len(self.k)
        k = sum(self.k)
        matr = []
        for i in range(n):
            arr = []
            v = 0
            for j in range(m):
                maxi = res[i][0+v]
                maxj = 0+v
                for f in range(1,self.k[j]):
                    if maxi< res[i][f+v]:
                        maxi = res[i][f+v]
                        maxj = f+v
                for f in range(self.k[j]):
                    if f+v == maxj:
                        arr.append(1)
                    else:
                        arr.append(0)
                v= v+self.k[j]
            matr.append(arr)
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        #print(matr)
        for i in range(n):
            for j in range(k):
                if matr[i][j] == 1 and Y[i][j] == 1:
                    TP = TP+1
                elif matr[i][j] == 0 and Y[i][j] == 0:
                    TN = TN+1
                elif matr[i][j]== 1 and Y[i][j] == 0:
                    FP = FP+1
                elif matr[i][j] == 0 and Y[i][j] == 1:
                    FN = FN+1
        #print('TP = ', TP, ', FP = ', FP, ', FN = ', FN, ', TN = ', TN)
        accuracy = (TP+TN)/(TP+FP+FN+TN)
        #precision = TP/(FP+TP)
        #recall = TP/(TP+FN)
        #f1_score = 2*precision*recall/(recall+precision)
        loss, accuracy = self.model.evaluate(X, Y, verbose=1)
        precision = precision_m(Y, res) 
        recall = recall_m(Y, res)
        f1_score = f1_m(Y, res)
        print ('accuracy = ',accuracy, 'precision = ', precision.numpy(), 'recall = ', recall.numpy(), 'f1_score = ', f1_score.numpy())
   
    def save_model(self):
        #self.model.save_weights('my_model_weights.h5')
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        model_lstm_save_path = 'Базовий.keras'
        self.model.save(model_lstm_save_path)
        
        
    def load_model(self):
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        #self.model.load_weights('my_model_weights.h5')
        self.model = load_model('Базовий.keras')
        
        
    def predict_m(self, X):
        self.k = [2, 2]
        #self.model.load_weights('my_model_weights.h5')
        #self.model = load_model('Базовий.h5')
        res = self.model.predict(X)
        print(res)
        n = len(res)
        m = len(self.k)
        matr = []
        for i in range(n):
            arr = []
            v = 0
            for j in range(m):
                maxi = res[i][0+v]
                maxj = 0+v
                for f in range(1,self.k[j]):
                    if maxi< res[i][f+v]:
                        maxi = res[i][f+v]
                        maxj = f+v
                for f in range(self.k[j]):
                    if f+v == maxj:
                        arr.append(1)
                    else:
                        arr.append(0)
                v= v+self.k[j]
            matr.append(arr)
        print(matr)


if __name__ == '__main__':


    fname2 = 'Ноутбук1.xlsx'
    fname1 = 'Смартфон.xlsx'
    fnames = [fname1, fname2]  
    cl = Learning()
    cl.creat_model()
    X, Y = cl.creat_x_y(fnames) 
    fname3 = ['test.xlsx']  
    x, y = cl.creat_x_y(fname3)
    history = cl.learn_model(X, Y, (x, y))
    
    
    
    cl.test_model(x, y)
    
    
    fname4 = ['Final_test.xlsx']  
    x1, y1 =  cl.creat_x_y(fname4)
    cl.test_model(x1, y1)  
    #cl.save_model() 
    
    
    text = ['Ноутбук не тягне ігри ', 'Крутий телефон', 'Ноутбук сподобався', 'Телефон з браком']
    x2 = cl.normalize(text)
    cl.predict_m(x2)


       
                        
                        
                        
                
                
                
        
