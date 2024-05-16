# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 18:01:23 2022

@author: E. Babenko

Program: Time series predict
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # computational pracesses on the processor

from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from xgboost import XGBRegressor
import shap
class Model ():
    
    def __init__(self):
        self.features_considered = ['HE','actual_load','temperature','heating', 'season', 'work_days','week_days_x',  'week_days_y']
        self.features_considered2 = ['actual_load', 'HE', 'temperature','heating', 'season', 'work_days', 'week_days_x', 'week_days_y']
        
    def get_date(self, date, season, holidays, workdays):
        self.date = date 
        self.season = season 
        self.holidays = holidays
        self.workdays = workdays
        
    
    
    def set_date(self, year, m, d, index):
        days_m = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if (year%400 == 0 or year%100 != 0) and year%4 == 0:
            days_m[1] = 29
        N = 0 # list number
        for i in range(m-1):
            N = N+days_m[i]*24
        if index == 0: # biginning of the day
            N = N+(d-1)*24
        elif index == 1: #end of the day
            N = N+d*24
        return N
    
    
    def add_columns(self, df, ss, hd, wd):
        days_m = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        n = 8760 #The amount of times in a not leap year 
        n1  = self.set_date(ss[0][0], ss[0][1], ss[0][2], 0)
        n2  = self.set_date(ss[1][0], ss[1][1], ss[1][2], 0)
        
        if (ss[0][0]%400 == 0 or ss[0][0]%100 != 0) and ss[0][0]%4 == 0:
            days_m[1] = 29
            n = 8784
        season = [1.5]*n
        for i in range(n1, n2):
            season[i] = -1.5
        start = str(ss[0][0])+'-01-01 00:00:00'
        end = str(ss[0][0])+'-12-31 23:00:00'
        t = pd.date_range(start=start, end=end, periods=n) # date list
        day_x = []
        day_y = []
        month = []
        work = [1]*n
        heating = [1]*n
        for i in range(n):
            iwd = t[i].isoweekday() # day of the week
            day_y.append(round(math.sin(iwd*2*math.pi/7), 4))
            day_x.append(round(math.cos(iwd*2*math.pi/7), 4))
            #day_x.append((iwd-4)/2)
            month.append(t[i].month)
            if iwd >5:
                work[i] = -1 # weekends
        for el in wd: #additional working days
            i = self.set_date(el[0], el[1], el[2], 0)
            for j in range(24):
                work[i+j] = el[3]
        for el in hd: #holidays
            i = self.set_date(el[0], el[1], el[2], 0)
            for j in range(24):
                work[i+j] = el[3]
        for i in range(n1, n1+720): # turning on the heater 
            if df.loc[i, 'temperature'] < 15:
                heating[i] = (df.loc[i, 'temperature']-15)/15
        for i in range(n2-720, n2):
            if df.loc[i, 'temperature'] < 15:
                heating[i] = (df.loc[i, 'temperature']-15)/15
        df['heating'] = heating
        df['week_days_x'] = day_x 
        df['week_days_y'] = day_y
        #df['month'] = month
        df['season'] = season
        df['work_days'] = work
        
        return df
    
   
    
    def concat_data(self, pdt, fns, end):
        
        data = pd.DataFrame()
        n = [0]
        P = 2*7*24 #interval for training beafore and after the set
        p = []
        from datetime import date
        n_day = date(self.date[0][0], self.date[0][1], self.date[0][2]).isoweekday()
        for i in range(len(fns)):
            excel = pd.ExcelFile(fns[i])
            sheets = excel.sheet_names
            df = excel.parse(sheets[0])
            e_day = date(self.season[i][0][0], self.date[0][1], self.date[0][2]).isoweekday()
            p.append(P+(n_day-e_day)*24)
            df = self.add_columns(df, self.season[i], self.holidays[i], self.workdays[i])
            n.append(df.shape[0])
            d = pd.DataFrame()
            for el in self.features_considered:
                i = df.columns.get_loc(el)
                d[el] = np.array(df.values[:, i], dtype=int)
            data = pd.concat([data, d],ignore_index=True)
        
        d = pd.concat([data, pdt], ignore_index=True)
        
        v = d.shape[0]
        N = 8760 
        if (self.date[0][0]%400 == 0 or self.date[0][0]%100 != 0) and self.date[0][0]%4 == 0:
            N = 8784
        z = N-end-1
        d = d.values[:v-z, :3]       
        d_mean = d.mean(axis=0)
        d_std =  d.std(axis=0)
        val_dt = data.values[:, :3]
        #data normalisation
        data.iloc[:, :3] = (val_dt - d_mean)/d_std
        dt = data.values
        val_pdt = pdt.values[:end, :3]
        pdt.iloc[:end, :3] = (val_pdt - d_mean)/d_std
        pdt = pdt.values 
        #taking data at the gift interval
        data_train = []
        for i in range(len(n)-1): 
            start = self.set_date(self.season[i][0][0], self.date[0][1], self.date[0][2], 0)+n[i]
            end = self.set_date(self.season[i][0][0], self.date[1][1], self.date[1][2], 1)+n[i]       
            if start-p[i]> 0:
                start = start - p[i] 
            else:
                start = 0
            if end +p[i] <= n[i+1]-1:
                end = end +p[i] 
            else:
                end = n[i+1]-1
            for j in range(start, end):
                data_train.append(dt[j][:])
        return np.array(data_train), pdt, d_mean, d_std
            
    
    
    def create_time_steps(self, length):
        return list(range(-length, 0))
    
    
    def multi_step_plot(self, history, true_future, prediction):
        plt.figure(figsize=(12, 6))
        n1 = len(history)
        n2 = len(true_future)
        d_e1 = datetime(self.date[0][0], self.date[0][1], self.date[0][2], 0)
        d_s1 = d_e1- timedelta(hours = n1)
        d_s2 = datetime(self.date[0][0], self.date[0][1], self.date[0][2], 0)
        d_e2 = d_s2 + timedelta(hours = n2)
        #X-axis 
        num_in = pd.date_range(start=d_s1, end=d_e1, periods=n1) 
        num_out = pd.date_range(start=d_s2, end=d_e2, periods=n2)
        
        plt.plot(num_in, history, label='History')
        plt.plot(num_out, true_future, 'b',
                   label='True Future')
        if prediction.any():
            plt.plot(num_out, prediction, 'r',
                     label='Predicted Future')
        plt.legend(loc='upper left')
        plt.show()
    
    
    def x_data(self, dataset, start_index, end_index, history_size, target_size):
        data = []
        
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
        
        for i in range(start_index, end_index):
            indices = range(i-history_size, i)
            data.append(dataset[indices])
        
        return np.array(data)
    
    
    def y_data(self, target, start_index, end_index, history_size, target_size):
        
        labels = []
        
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(target) - target_size
        
        for i in range(start_index, end_index):
            labels.append(target[i:i+target_size])
        
        return np.array(labels)
    
    
    
    def x_data_2(self, dataset, start_index, end_index, history_size,target_size):
        data = []
        
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
        
        for i in range(start_index, end_index, target_size):
            indices = range(i-history_size, i)
            data.append(dataset[indices])
        
        return np.array(data)
    
    
    def y_data_2(self, target, start_index, end_index, history_size, target_size):
        
        labels = []
        
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(target) - target_size
        
        for i in range(start_index, end_index, target_size):
            labels.append(target[i:i+target_size])
        
        return np.array(labels)
    
    
    def x_data2(self, dataset, start_index, end_index, target_size):
        data = []
        
        #start_index = start_index + history_size
        end_index=end_index-target_size+1
        for i in range(start_index, end_index):
            indices = range(i, i+target_size)
            data.append(dataset[indices])
        
        return np.array(data)
    
    
    def y_data2(self, target, start_index, end_index, target_size):
        
        labels = []
        
        end_index=end_index-target_size+1
        for i in range(start_index, end_index):
            indices = range(i, i+target_size)
            labels.append(target[indices])
        
        return np.array(labels)
    
    
     
    def forecast_evalution (self, val, pred):
         error = []
         n  = len(val)
         MAPE = 0
         for i in range(n):
             er = abs(val[i]-pred[i])
             error.append(er)
             MAPE = MAPE + abs(er/val[i])
         MAPE = MAPE/n
         plt.plot(error)
         plt.title('MAPE: %.4f'% MAPE)
         plt.show()  
         
    
    def predict(self, multi_step_model, x_val, future_target, d_std, d_mean, k0, k1, I, v, r):
        pr = np.array([])
        pr2 = np.array([])
        l = 0
        for i in range(0,v, future_target):
            st = l
            if l > k1:
                st = k1
            for j in range(st, 0, -1):
                x_val[i][k1-j-1][I] = pr[l-j]
            x = np.array([x_val[i]])
            predict = multi_step_model.predict(x)[0]
            pr = np.append(pr, predict)
            for j in range(future_target):
                predict[j] = predict[j]*d_std[I] +d_mean[I]
            pr2 = np.append(pr2, predict)
            l = l+future_target
        if r != future_target:
            for j in range(r, 0, -1):
                x_val[k0-1][k1-j-1][I] = pr[l-j]
            x = np.array([x_val[k0-1]])
            predict =  multi_step_model.predict(x)[0]
            for j in range(future_target):
                predict[j] = predict[j]*d_std[I] +d_mean[I]
            pr2 = np.append(pr2[:r], predict[future_target-1])
        return pr2
    
    def predict_2(self, multi_step_model, x_val, future_target, d_std, d_mean, k0, k1, I, v, r):
        pr = np.array([])
        pr2 = np.array([])
        l = 0
        for i in range(0, k0):
            st = l
            if l > k1:
                st = k1
            for j in range(st, 0, -1):
                x_val[i][k1-j-1][I] = pr[l-j]
            x = np.array([x_val[i]])
            predict = multi_step_model.predict(x)[0]
            pr = np.append(pr, predict)
            for j in range(future_target):
                predict[j] = predict[j]*d_std[I] +d_mean[I]
            pr2 = np.append(pr2, predict)
            l = l+future_target
        if r != future_target:
            for j in range(r, 0, -1):
                x_val[k0-1][k1-j-1][I] = pr[l-j]
            x = np.array([x_val[k0-1]])
            predict =  multi_step_model.predict(x)[0]
            for j in range(future_target):
                predict[j] = predict[j]*d_std[I] +d_mean[I]
            pr2 = np.append(pr2[:r], predict[future_target-1])
        return pr2
    
    
    
    def multi_step_multidimensional (self,df, fns): # forcest based on datain the past period 
        
        start = self.set_date(self.date[0][0], self.date[0][1], self.date[0][2], 0) 
        end = self.set_date(self.date[1][0], self.date[1][1], self.date[1][2], 1)
        I = 1 # number in the list actual_load
        future_target = 4
        past_history = 4*24 
        BATCH_SIZE = 50
        BUFFER_SIZE = 5000
        EVALUATION_INTERVAL = 50
        EPOCHS = 18
        print(EPOCHS, future_target, past_history)
        df = self.add_columns(df, self.season[2], self.holidays[2], self.workdays[2])
        df = df[self.features_considered]
        y_test = df.values[start:end, I]
        hist = df.values[start-7*24:start, I]
        dt, pdt, d_mean, d_std = self.concat_data(df, fns, end)
        n = dt.shape[0]
                             
        y_train = self.y_data(dt[:, I], 0, n-future_target, past_history,future_target)
        x_train = self.x_data(dt[:, :], 0,n-future_target, past_history,future_target)                                                        
        train_data = Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        
        start_test = start - past_history
       
        multi_step_model = Sequential()
        multi_step_model.add(LSTM(64,
                                  return_sequences=True,
                                  input_shape=x_train.shape[-2:]))
        multi_step_model.add(LSTM(16, activation='relu'))
        multi_step_model.add(Dense(future_target))

        multi_step_model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae')
        multi_step_history = multi_step_model.fit(train_data, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL)
        

        x_test = self.x_data(pdt[:, :], start_test, end, past_history, future_target)
                                  
        k0 = x_test.shape[0]
        k1 = x_test.shape[1]
        v = end-start
        r = future_target - (math.ceil(v/future_target))*future_target + v #remaider. if the length of the entire forcast is not a multiple of the forcast interval
        pr2 = self.predict(multi_step_model, x_test, future_target, d_std, d_mean, k0, k1, I, v, r)

        self.forecast_evalution(y_test, pr2)
        self.multi_step_plot(hist, y_test, pr2)
        
    
        
    def multi_step_multidimensional2 (self,df, fns):# forcest based on raging data
        
        start = self.set_date(self.date[0][0], self.date[0][1], self.date[0][2], 0) 
        end = self.set_date(self.date[1][0], self.date[1][1], self.date[1][2], 1)
        I = 0 # number in the list actual_load
        future_target = 7*24 
        BATCH_SIZE = 50
        BUFFER_SIZE = 5000
        EVALUATION_INTERVAL = 50
        EPOCHS = 24
        print(EPOCHS, future_target)
        df = self.add_columns(df, self.season[2], self.holidays[2], self.workdays[2])
        df = df[self.features_considered]
        y_test = df.values[start:end, I]
        hist = df.values[start-7*24:start, I]
        dt, pdt, d_mean, d_std = self.concat_data(df, fns, end)
        n = dt.shape[0]
        
        y_train = self.y_data2(dt[:, I], 0, n, future_target)
        x_train = self.x_data2(dt[:, 1:], 0,n, future_target)                                                        
        train_data = Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
       
        multi_step_model = Sequential()
        multi_step_model.add(LSTM(64,
                                  return_sequences=True,
                                  input_shape=x_train.shape[-2:]))
        multi_step_model.add(LSTM(16, activation='relu'))
        multi_step_model.add(Dense(future_target))

        multi_step_model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae')
        multi_step_history = multi_step_model.fit(train_data, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL)

        x_test = self.x_data2(pdt[:, 1:], start, end, future_target)
                                             
                                            
        k0 = x_test.shape[0]
        x = np.array([x_test[0]])
        v = end-start
        pr = np.array([])
        r = future_target - (math.ceil(v/future_target))*future_target + v
        for i in range(0, k0, future_target):
            x = np.array([x_test[i]])
            predict = multi_step_model.predict(x)[0]
            for j in range(future_target):
                predict[j] = predict[j]*d_std[I] +d_mean[I]
            pr = np.append(pr, predict)
        if r != future_target:
            x = np.array([x_test[k0-1]])
            predict = multi_step_model.predict(x)[0]
            for j in range(r):
                predict[j] = predict[j]*d_std[I] +d_mean[I]
            pr = np.append(pr, predict[:r])
   
        self.forecast_evalution(y_test, pr)
        self.multi_step_plot(hist, y_test, pr)
    
    
    
        
    def concat_data_xgb(self, fns, dt, start):
        
        data = pd.DataFrame()
        for i in range(len(fns)):
            excel = pd.ExcelFile(fns[i])
            sheets = excel.sheet_names
            df = excel.parse(sheets[0])
            df = self.add_columns(df, self.season[i], self.holidays[i], self.workdays[i])
            d = pd.DataFrame()
            for el in self.features_considered:
                i = df.columns.get_loc(el)
                d[el] = np.array(df.values[:, i], dtype=int)
            data = pd.concat([data, d],ignore_index=True)

        data = pd.concat([data, dt.iloc[:start, :]], ignore_index=True) 
        return data.values
    
    
    def concat_data_xgb_add(self, fns, dt, start, I):
        
        data = pd.DataFrame()
        day = []
        week = []
        #year = []
        for i in range(len(fns)):
            excel = pd.ExcelFile(fns[i])
            sheets = excel.sheet_names
            df = excel.parse(sheets[0])
            df = self.add_columns(df, self.season[i], self.holidays[i], self.workdays[i])
            d = pd.DataFrame()
            for el in self.features_considered2:
                i = df.columns.get_loc(el)
                d[el] = np.array(df.values[:, i], dtype=int)
            data = pd.concat([data, d],ignore_index=True)
        
        dd = pd.concat([data, dt.iloc[:, :]], ignore_index=True)
        data = pd.concat([data, dt.iloc[:start, :]], ignore_index=True)
        n = data.shape[0]
        '''
        for i in range(n-8760):
            year.append(sum(dd.iloc[i:i+8760, I]))
        for i in range(8592, n-168):
            week.append(sum(dd.iloc[i:i+168, I]))
        for i in range(8736, n-24):
            day.append(sum(dd.iloc[i:i+24, I]))
        '''
        for i in range(n-168):
            week.append(sum(dd.iloc[i:i+168, I]))
        
        for i in range(144, n-24):
            day.append(sum(dd.iloc[i:i+24, I]))
        data = data.iloc[168:, :]
        data['sum_day'] = day
        #data['sum_week'] = week
        #data['sum_year'] = year
        n = data.shape[0]
        return data, np.array([data.values[n-1, :]]), dd.values
    

    
    def model_xgboost(self, df, fns):
        I = 0 # number in the list actual_load
        start = self.set_date(self.date[0][0], self.date[0][1], self.date[0][2], 0) 
        end = self.set_date(self.date[1][0], self.date[1][1], self.date[1][2], 1)
        df = self.add_columns(df, self.season[2], self.holidays[2], self.workdays[2])
        df = df[self.features_considered]
        hist = df.values[start-7*24:start, I]
        y_test = df.values[start:end, I]
        dt = self.concat_data_xgb(fns, df, start)
        n = dt.shape[0]
        x_train = dt[:n-1, :]
        y_train = dt[1:, I]
        model = XGBRegressor()
        model.fit(x_train, y_train,eval_set=[(x_train, y_train)],
        early_stopping_rounds=50, verbose=False)
        x_test = df.values[start-1:end-1, :]
        predict = []
        for i in range(end-start):
            x = np.array([x_test[i]])
            pred = model.predict(x)
            predict.append(pred)
            if i <end-start-1:
                x_test[i+1,I] = pred
        predict = np.array(predict)
        self.forecast_evalution(y_test, predict)
        self.multi_step_plot(hist, y_test, predict)
        
        
    def model_xgboost2(self, df, fns): #model with additional parameters
        I = 0 # number in the list actual_load
        start = self.set_date(self.date[0][0], self.date[0][1], self.date[0][2], 0) 
        end = self.set_date(self.date[1][0], self.date[1][1], self.date[1][2], 1)
        df = self.add_columns(df, self.season[2], self.holidays[2], self.workdays[2])
        df = df[self.features_considered2]
        data = df.values
        hist = data[start-7*24:start, I]
        y_test = data[start:end, I]
        data_tr, x_test, dd = self.concat_data_xgb_add(fns, df, start, I)
        n = data_tr.shape[0]
        dt = data_tr.values[:n, :]
        x_train = dt[:n-1, :]
        y_train = dt[1:, I]
        model = XGBRegressor()
        model.fit(x_train, y_train,eval_set=[(x_train, y_train)],
                  early_stopping_rounds=50, verbose=False)
        predict = []
        for i in range(start, end):
            pred = model.predict(x_test)
            predict.append(pred)
            x_test[0,I] = pred
            for j in range(1, 8):
                x_test[0,j] = data[i, j]
            dd[i+17544, I] = pred 
            x_test[0,8] = sum(dd[i+17520:i+17544, I]) #day
            #x_test[0,9] = sum(dd[i+17376:i+17544, I]) #week
            #x_test[0,10] = sum(dd[i+8784:i+17544, I]) #year
        predict = np.array(predict)
        self.forecast_evalution( y_test, predict)
        self.multi_step_plot(hist, y_test, predict)
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(data_tr[:50])
        shap.summary_plot(shap_values, data_tr[:50])
    
        
        


mod = Model()
fns = ['2_2015 - Исторические данные.xlsx', '2_2016 - Исторические данные.xlsx']

workdays = [[[2015, 1, 17, 1], [2015, 1, 31, 1], [2015, 2, 14, 1]], 
           [[2016, 1, 16, 1], [2016, 3, 12, 1], [2016, 7, 2, 1]],
           [[2017, 5, 13, 1], [2017, 8, 19, 1]]]


holidays = [[[2015, 1, 1, -2], [2015, 1, 2, -1], [2015, 1, 7, -1], [2015, 1, 8, -1], 
             [2015, 1, 9, -1], [2015, 3, 9, -1],  [2015, 4, 12, -2], [2015, 4, 13, -1],
             [2015, 5, 1, -1], [2015, 5, 4, -1], [2015, 5, 11, -1], [2015, 6, 1, -1],
             [2015, 6, 29, -1], [2015, 8, 24, -1], [2015, 10, 14, -1]],
             
            [[2016, 1, 1, -2], [2016, 1, 7, -1], [2016, 1, 8, -1], [2016, 3, 7, -1], 
             [2016, 3, 8, -1],  [2016, 5, 1, -2], [2016, 5, 2, -1], [2016, 5, 3, -1], 
             [2016, 5, 9, -1],  [2016, 6, 20, -1], [2016, 6, 27, -1], [2016, 6, 28, -1], 
             [2016, 8, 24, -1], [2016, 10, 14, -1]],

            [[2017, 1, 1, -2],  [2017, 1, 2, -1], [2017, 1, 7, -1], [2017, 1, 9, -1], 
            [2017, 3, 8, -1], [2017, 4, 16, -2], [2017, 4, 17, -1], [2017, 5, 1, -1], 
            [2017, 5, 2, -1], [2017, 5, 8, -1], [2017, 5, 9, -1], [2017, 6, 5, -1],
            [2017, 6, 28, -1],[2017, 8, 24, -1], [2017, 8, 25, -1],
            [2017, 10, 16, -1]]]


holidays2 = [[[2015, 1, 1, -1], [2015, 1, 2, -0.75], [2015, 1, 7, -0.75], [2015, 1, 8, -0.75], 
             [2015, 1, 9, -0.75], [2015, 3, 9, -0.75],  [2015, 4, 12, -1], [2015, 4, 13, -0.75],
             [2015, 5, 1, -1], [2015, 5, 4, -0.8], [2015, 5, 11, -0.5], [2015, 6, 1, -0.5],
             [2015, 6, 29, -0.75], [2015, 8, 24, -0.75], [2015, 10, 14, -0.75]],
             
            [[2016, 1, 1, -1], [2016, 1, 7, -0.75], [2016, 1, 8, -0.75], [2016, 3, 7, -0.75], 
             [2016, 3, 8, -0.75],  [2016, 5, 1, -1], [2016, 5, 2, -0.8], [2016, 5, 3, -0.75], 
             [2016, 5, 9, -0.75],  [2016, 6, 20, -0.75], [2016, 6, 27, -0.75], [2016, 6, 28, -0.75], 
             [2016, 8, 24, -0.75], [2016, 10, 14, -0.75]],

            [[2017, 1, 1, -1],  [2017, 1, 2, -0.75], [2017, 1, 7, -0.75], [2017, 1, 9, -0.75], 
            [2017, 3, 8, -0.75], [2017, 4, 16, -1], [2017, 4, 17, -0.75], [2017, 5, 1, -0.75], 
            [2017, 5, 2, -0.75], [2017, 5, 8, -0.75], [2017, 5, 9, -0.75], [2017, 6, 5, -0.75],
            [2017, 6, 28, -0.75],[2017, 8, 24, -0.75], [2017, 8, 25, -0.5],
            [2017, 10, 16, -0.75]]]


date = [[2017, 4, 15], [2017, 4, 22]] 

season = [[[2015, 4, 10], [2015, 10, 12]], 
          [[2016, 4, 11], [2016, 10, 14]], 
          [[2017, 4, 17], [2017, 10, 23]]]     
fn1 = '2_2017 - Исторические данные.xlsx'
excel = pd.ExcelFile(fn1)
sheets = excel.sheet_names
df = excel.parse(sheets[0])
#print(df.loc[:10, 'actual_load'])
mod.get_date(date, season, holidays, workdays)
#mod.multi_step_multidimensional(df, fns) 

mod.model_xgboost2(df, fns)