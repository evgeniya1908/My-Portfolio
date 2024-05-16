# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:38:19 2020

@author: Admin
"""
#from fbs_runtime.application_context.PyQt5 import ApplicationContext
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QTableWidgetItem, QApplication, QMainWindow, QFileDialog, QMessageBox)
import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, Qt
import inspect, os.path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
#from Crypto.Cipher import DES
#import time
import random
from cryptography.fernet import Fernet
   
def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def is_int(str):
    try:
        int(str)
        return True
    except ValueError:
        return False

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        Id = [i for i in range(1, data.shape[0] + 1)]
        self._data = pd.DataFrame()
        self._data['Id'] = Id
        liss = list(data)
        for i in range (data.shape[1]):
            self._data[liss[i]] = data.values[:,i] 
        
        
    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None



class FileData(QMainWindow):

    def __init__(self):
        super().__init__()
        self._fname = "" # ім'я файлу
        self._data = pd.DataFrame() # таблиця даних
        self._N = 0 # кількість строк
        self._M = 0 # кількість стовпців
        
     
    def showDialog(self):
        fn = QFileDialog.getOpenFileName(self, 'Open file', None, "File (*.txt *.xlsx *.parquet)")
        self._fname = "".join(fn[0]) 

        if fn[0]:
            try:
                if self._fname.endswith(".txt") == True:
                    self._data = pd.read_csv(self._fname, sep = '\t')

                elif self._fname.endswith(".xlsx") == True:
                    self._data = pd.read_excel(self._fname)
                
                elif self._fname.endswith(".parquet") == True:
                    self._data = pd.read_parquet(self._fname)
                
                else:
                    QMessageBox.critical(self, "Помилка ", "Неможливо відкрити файл", QMessageBox.Ok)
            except:
                QMessageBox.critical(self, "Помилка ", "Файл не містить таблицю", QMessageBox.Ok)
        else:
            QMessageBox.critical(self, "Помилка ", "Не має файлу", QMessageBox.Ok)
        
        if  self._data.shape[1] > 0:
            self._N = self._data.shape[0]
            self._M = self._data.shape[1]
        else:
            QMessageBox.critical(self, "Помилка ", "Немає таблиці даних у файлу", QMessageBox.Ok)
      
        
    def openParquet(self, fn):
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path     = os.path.dirname(os.path.abspath(filename))
        self._fname = path + "\\" + "~" + fn + ".parquet"
        if os.path.isfile(self._fname) == True:
            self._data = pd.read_parquet(self._fname)
            self._N = self._data.shape[0]
            self._M = self._data.shape[1]
        else:
            QMessageBox.critical(self, "Помилка ", "Не має файлу", QMessageBox.Ok)

    
    def convertNewData (self, dt):
        head = list(self._data)
        n = np.shape(dt)[0]
        m = self._M
        data =  pd.DataFrame()
        if self._N > 0: 
            try:
                for i in range(n):
                    for j in range(m):
                        if self._data.dtypes[j]:
                            dt[i][j] = int(dt[i][j])
                        elif self._data.dtypes[j] == 'float64':
                            dt[i][j] = float(dt[i][j])
                        elif self._data.dtypes[j] == 'bool':
                            dt[i][j] = bool(dt[i][j])
                data = pd.DataFrame(dt, columns=head)
            except:
                QMessageBox.critical(self, "Помилка ", "Введені дані неспівпадають з типом", QMessageBox.Ok)
        else:
            for i in range(n):
                for j in range(m):
                    st = dt[i][j]
                    if is_int(st) == True:
                        dt[i][j] = int(dt[i][j])
                    elif is_float(dt[i][j]) == True:
                        dt[i][j] = float(st)
                    elif dt[i][j] == 'true' or dt[i][j] == 'True':
                        dt[i][j] = True
                    elif dt[i][j] == 'false' or dt[i][j] == 'False':
                        dt[i][j] = False
            data = pd.DataFrame(dt, columns=head)
        return data
    
    
    def cleanFile(self):
        head = list(self._data)
        arr = []
        self._data = pd.DataFrame(arr, columns=head)
        self._data.to_parquet(self._fname, index = False)   
    
    def saveNewData (self, dt):
        try:
            self._data = pd.concat([self._data, dt], ignore_index=True)
            filename = inspect.getframeinfo(inspect.currentframe()).filename
            path     = os.path.dirname(os.path.abspath(filename))
            name = os.path.basename(self._fname)
            n = name.split('.')
            if n[0].startswith('~') == True:
                self._data.to_parquet(self._fname, index = False)
            else:
                self._fname = path + "\\" + "~" + n[0] + ".parquet"
                self._data.to_parquet(self._fname, index = False) 
                subprocess.call(['attrib', '+h', self._fname])
            self._N = self._data.shape[0]
        except:
             QMessageBox.critical(self, "Помилка ", "Файл перезаписався", QMessageBox.Ok)    
        
        
    
    def creatFileData(self, fn):
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path     = os.path.dirname(os.path.abspath(filename))
        self._fname = path + "\\"+ "~" + fn + ".parquet"
        self._data.to_parquet(self._fname, index = False)
        #subprocess.call(['attrib', '+h', self._fname])



class Data (FileData):
    
    def __init__(self):
        super().__init__()
        self._norm = pd.DataFrame()
        self._fileCluster = ''
        self._fileW = ''
        self._path = ''
        self._cluster = []
        self._nv = 0 #кількість старих записів
        self._class = []
        
    def pathData(self):
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        self._path = os.path.dirname(os.path.abspath(filename))
    
    
    def dataNorm(self):
        mini = 0
        maxi = 0
        for i in range(self._N):
            for j in range(self._M):
                if mini > self._data.values[i,j]:
                    mini = self._data.values[i,j]
                if maxi < self._data.values[i,j]:
                    maxi = self._data.values[i,j]
        if mini == 0 and maxi == 1:
            self._norm = self._data
        else:
            QMessageBox.critical(self, "Помилка ", "Треба провести нормалізацію даних", QMessageBox.Ok)

        
    def transform(self):
        self._norm = pd.DataFrame()
        for i in range(self._M): 
            norm = []
            maxi =  max(self._data.values[:,i])
            mini = min(self._data.values[:, i])
            for j in range(self._N):
                n =(self._data.values[j,i]-mini)/(maxi-mini)
                norm.append(round(n,4))
            self._norm[str(i)] = norm

        
             
class Kohonen(Data):
   
    def __init__(self):
        super().__init__()
        self._K = 1 #кількість нейронів
        self._w = []
        self._v = 0.3
        
        
    def setNameFileClusterNNK1(self):
        self.pathData()
        name = os.path.basename(self._fname)
        n = name.split('.')
        if n[0].startswith('~') == True:
            self._fileCluster = self._path + "\\" + n[0] + '_clusterNNK1.parquet'
        else:
            self._fileCluster = self._path + "\\" + "~" + n[0] + '_clusterNNK1.parquet'
        
    
    def setNameFileClusterNNK2(self):
        self.pathData()
        name = os.path.basename(self._fname)
        n = name.split('.')
        if n[0].startswith('~') == True:
            self._fileCluster = self._path + "\\" + n[0] + '_clusterNNK2.parquet'
        else:
            self._fileCluster = self._path + "\\" + "~" + n[0] + '_clusterNNK2.parquet'
    

    def setNameFileWeight1(self):
        self.pathData()
        name = os.path.basename(self._fname)
        n = name.split('.')
        if n[0].startswith('~') == True:
            self._fileW = self._path + "\\" + n[0] + '_weightNNK1.parquet'
        else:
            self._fileW = self._path + "\\" + "~" + n[0] + '_weightNNK1.parquet'
              
        
    def setNameFileWeight2(self):
        self.pathData()
        name = os.path.basename(self._fname)
        n = name.split('.')
        if n[0].startswith('~') == True:
            self._fileW = self._path + "\\" + n[0] + '_weightNNK2.parquet'
        else:
            self._fileW = self._path + "\\" + "~" + n[0] + '_weightNNK2.parquet'
            
        
   
    def initWeight1(self):
        min_w = 0.5-1/(self._M**0.5) 
        max_w = 0.5+1/(self._M**0.5)
        for i in range(self._K):
            arr = []
            for j in range(self._M):
                arr.append(random.uniform(min_w,max_w)) #генерація вагів від 0 до 1
            self._w.append(arr)
    
    
    def initWeight2(self):
        self._w = [[0.694, 0.694, 0.694, 0.08, 0.55, 0.85], #0
                   [0.694, 0.694, 0.7, 0.04, 0.25, 0.9], #1
                   [0.75, 0.75, 0.75, 0.15, 0.62, 0.9], #2
                   [0.3, 0.3, 0.3, 0.25, 0.5, 0.5], #3
                   [0.694, 0.9, 0.3, 0.8, 0.5, 0.9], #4
                   [0.9, 0.25, 0.89, 0.04, 0.78, 0.9], #5
                   [0.0, 0.0, 0.0, 0.25, 0.5, 0.5], #6
                   [0.0, 0.0, 0.0, 0.0, 0.5, 0.0], #7
                   [0.0, 0.694, 0.0, 0.25, 0.5, 0.0], #8
                   [0.0, 0.694, 0.0, 0.25, 0.4, 0.5], #9
                   [0.0, 0.694, 0.0, 0.0, 0.5, 0.5], #10
                   [0.85, 0.85, 0.85, 0.0, 0.8, 0.5], #11
                   [0.694, 0.0, 0.86, 0.25, 0.75, 0.5], #12
                   [0.3, 0.694, 0.694, 0.25, 0.75, 0.5], #13
                   [0.694, 0.694, 0.694, 0.25, 0.75, 0.5], #14
                   [0.85, 0.694, 0.85, 0.25, 0.5, 0.5], #15
                   [0.694, 0.694, 0.85, 0.25, 0.5, 0.5], #16
                   [0.694, 0.694, 0.694, 0.02, 0.75, 0.9], #17
                   [0.694, 0.694, 0.694, 0.02, 0.75, 0.5], #18
                   [0.69, 0.69, 0.69, 0.001, 0.4, 0.9], #19
                   [0.694, 0.694, 0.85, 0.4, 0.8, 0.0], #20
                   [0.9, 0.9, 0.9, 0.25, 0.8, 0.9]] #21
    

    def class_definition (self, elem):
        if elem == 0 or elem == 2 or elem == 6 or elem == 7:
            self._class.append(0)
        elif elem == 3 or elem == 4 or elem == 11 or elem == 17:
            self._class.append(1)
        elif elem == 1 or elem == 5:
            self._class.append(3)
        else:
            self._class.append(2)
            
    
    def equidistant(self, nd):
        minR = 10
        minI = 0
        for i in range(self._K):
            r = 0
            for j in range(self._M):
                r = r + (nd[j]-self._w[i][j])**2
            r = (r)**0.5
            if r < minR:
                minR = r
                minI = i

        return minI
    
    
    def weightCorrection (self, nd):
        minI = self.equidistant(nd)
        self._nv = self._nv + 1
        if self._nv % 20 == 0:
            self._v = self._v - 0.05*self._nv/20
        elif self._nv/20 >= 6:
            self._v = 0     
        for j in range(self._M):
            self._w[minI][j] = self._w[minI][j] + self._v*(nd[j]-self._w[minI][j])
        
        return minI
    
        
    def to_analyzeNNK(self, kind, K = 0):
        #start = time.time()
        if kind == 1:
            self.setNameFileClusterNNK1()
            self.setNameFileWeight1()
            if os.path.isfile(self._fileW) == True:
                dt = pd.read_parquet(self._fileW)
                self._K = dt.shape[0]
                self._nv = sum(dt.values[:, self._M])                    
                self._N = self._N - self._nv
                if self._N < 1:
                    QMessageBox.critical(self, "Помилка ", "Не має нових даних", QMessageBox.Ok)
                else:
                    for i in range(self._K):
                        self._w.append = dt.values[i]
                    for i in range(self._nv, self._N+self._nv):
                        nd = self._norm.values[i, :]
                        winner =  self.weightCorrection(nd)
                        self._cluster.append(winner)
            else:
                self._K = K
                self._v = 0.3
                self._nv = 0
                self.initWeight1()
                for i in range(self._N):
                    nd = self._norm.values[i, :]
                    winner =  self.weightCorrection(nd)
                    self._cluster.append(winner)
                      
        elif kind == 2 and self._M == 6:
            self.setNameFileClusterNNK2()
            self.setNameFileWeight2()
            if os.path.isfile(self._fileW) == True:
                dt = pd.read_parquet(self._fileW)
                self._K = 4
                self._nv = int(sum(dt.values[:, self._M]))
                self._N = int(self._N - self._nv)
                if self._N < 1:
                    QMessageBox.critical(self, "Помилка ", "Не має нових даних", QMessageBox.Ok)
                else:
                    for i in range(self._K):
                        self._w.append(dt.values[i])
                    for i in range(self._nv, self._N + self._nv):
                        nd = self._norm.values[i, :]
                        winner =  self.weightCorrection(nd)
                        self.class_definition(winner)
            else:
                self._K = 22
                self._v = 0.3
                self._nv = 0
                self.initWeight2()
                for i in range(self._N):
                    nd = self._norm.values[i, :]
                    winner =  self.weightCorrection(nd)
                    self._cluster.append(winner)
                    self.class_definition(winner)
        elif kind == 2 and self._M != 6:
            QMessageBox.critical(self, "Помилка ", "Недостатня кількість стовпців", QMessageBox.Ok)
        else: 
            print("Помилка. Цифра повинна бути 1 або 2")
       # print(time.time() - start)


class ART (Data):
    
    def __init__(self):
        super().__init__()
        self._R = 0.96
        self._v = 0.5
        self._w = []
    
    
    def setNameFileClusterART(self):
        self.pathData()
        name = os.path.basename(self._fname)
        n = name.split('.')
        if n[0].startswith('~') == True:
            self._fileCluster = self._path + "\\" + n[0] + '_clusterART.parquet'
        else:
            self._fileCluster = self._path + "\\" + "~" + n[0] + '_clusterART.parquet'
        
 
    def setNameFileWeightART(self):
        self.pathData()
        name = os.path.basename(self._fname)
        n = name.split('.')
        if n[0].startswith('~') == True:
            self._fileCluster = self._path + "\\" + n[0] + '_weightART.parquet'
        else:
            self._fileCluster = self._path + "\\" + "~" + n[0] + '_weightART.parquet'
        
  
    def rationing (self, norm):
        sumsq = 0
        arr = []
        for elem in norm:
            sumsq = sumsq + elem**2
        sumsq = sumsq**0.5
        if sumsq == 0:
            sumsq = 1
        for i in range(self._M):
            n = round(norm[i]/sumsq,4)
            arr.append(n)
        return arr
      
        
    def similarity (self, norm):
        nor = self.rationing(norm) 
        K = np.shape(self._w)[0]
        maxR = 0
        maxI = 0 
        for i in range(K):
            r = 0
            for j in range(self._M):
                r = r + self._w[i][j]*nor[j]
            if maxR < r:
                maxR = r
                maxI = i
        if maxR < self._R:
            self._w.append(nor)
            maxI = K
        else:
            for i in range(self._M):
                self._w[maxI][i] = (1 - self._v)*self._w[maxI][i] + self._v*nor[i]
        return maxI
    
       
    def to_analyzeART (self):
        #star = time.time()
        self.setNameFileClusterART()
        self.setNameFileWeightART()
        if os.path.isfile(self._fileW) == True:
            dt = pd.read_parquet(self._fileW)
            self._K = dt.shape[0]
            self._nv = sum(dt.values[:, self._M])                    
            self._N = int(self._N - self._nv)
            if self._N == 0:
                QMessageBox.critical(self, "Помилка ", "Не має нових даних", QMessageBox.Ok)
            else:
                for i in range(self._K):
                    for j in range(self._M):
                        self._w[i][j] = dt.values[i][j]
                for i in range(self._nv, self._N+self._nv):
                    nd = self._norm.values[i, :]
                    winner =  self.similarity(nd)
                    self._cluster.append(winner)
        else:
            self._nv = 0
            norm = self.rationing(self._norm.values[0])
            self._w.append(norm)
            self._cluster.append(0)
            for i in range(1, self._N):
                nd = self._norm.values[i, :]
                winner =  self.similarity(nd)
                self._cluster.append(winner)
                      
       # print(time.time()-star)

class EML(Data):
    
    def __init__(self):
        super().__init__()
        self._wout = np.array([])
        self._y = np.array([])
        self._win = np.array([])
        self._x = np.array([])

    def setNameFileClusterEML(self):
        self.pathData()
        name = os.path.basename(self._fname)
        n = name.split('.')
        if n[0].startswith('~') == True:
            self._fileCluster = self._path + "\\" + n[0] + '_clusterEML.parquet'
        else:
            self._fileCluster = self._path + "\\" + "~" + n[0] + '_clusterEML.parquet'
        

    def setNameFileWeightEML(self):
        self.pathData()
        name = os.path.basename(self._fname)
        n = name.split('.')
        if n[0].startswith('~') == True:
            self._fileCluster = self._path + "\\" + n[0] + '_weightEML.parquet'
        else:
            self._fileCluster = self._path + "\\" + "~" + n[0] + '_weightEML.parquet'
    

    def class_definition (self, elem):
        if elem == 0 or elem == 2 or elem == 6 or elem == 7:
            self._class.append(0)
        elif elem == 3 or elem == 4 or elem == 11 or elem == 17:
            self._class.append(1)
        elif elem == 1 or elem == 5:
            self._class.append(3)
        else:
            self._class.append(2)
       
            
    def setData (self):
        self.pathData()
        name = os.path.basename(self._fname)
        n = name.split('.')
        fl = self._path + "\\" + "~" + n[0] +'_clusterNNK2.parquet'
        train = pd.read_parquet(fl)
        lab = train.iloc[self._nv :, 6].values.astype('int32')
        self._x = train.iloc[:, :6].values.astype('float32')
        CLASSES = 4
        self._y = np.zeros([lab.shape[0], CLASSES])
        for i in range(lab.shape[0]):
            self._y[i][lab[i]] = 1
        self._y.view(type=np.matrix)
    
    def hidden_layer(self, x):
        a = np.dot(x, self._win)
        a = np.maximum(a, 0, a) # ReLU
        return a
    
    
    def predict(self, x):
        x = self.hidden_layer(x)
        y = np.dot(x, self._wout)
        return y   
    
    
    def EML(self):
        
        self.setData()
        self._win = np.random.random(size=[6, 4])
        X = self.hidden_layer(self._x)
        Xt = np.transpose(X)
        self._wout = np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, self._y))
        y = self.predict(self._norm)
        total = y.shape[0]
        for i in range(total):
             pr = np.argmax(y[i])
             self._class.append(pr)
             #self.class_definition(pr)
        self._cluster = pr
         
            
    def to_analyzeEML(self):
        #start = time.time()
        if self._norm.empty == True:
            QMessageBox.critical(self, "Помилка ", "Необхідно нормалізувати дані", QMessageBox.Ok)
        else:
            self.setNameFileClusterEML()
            self.setNameFileWeightEML()
            if os.path.isfile(self._fileW) == True:
                dt = pd.read_parquet(self._fileW)
                self._nv = sum(dt.values[:, 1])                    
                self._N = int(self._N - self._nv)
                if self._N == 0:
                    QMessageBox.critical(self, "Помилка ", "Не має нових даних", QMessageBox.Ok)
                else:
                    self.EML()
            else:
                self.EML()
        #print(time.time() - start)
                

class Result(Kohonen, ART, EML):
    
    def __init__(self):
        super().__init__()
        self._resCl = pd.DataFrame()
        self._resW = pd.DataFrame()
        self._num = []
      
        
    def resultClustering(self):
        if not self._cluster:
            QMessageBox.critical(self, "Помилка ", "Не має даних", QMessageBox.Ok)
        else:
            if self._nv == 0:
                self._resCl = self._norm
            else:
                for i in range(self._M):
                    arr = []
                    for j in range(self._N):
                        arr.append(self._norm.values[j,i])
                    self._resCl[str(i)] = arr
            if not self._class:
                self._resCl['Кластер'] = self._cluster
            else:
                self._resCl['Кластер'] = self._class
        
        
    def resultWeight (self):
        if not self._cluster:
            QMessageBox.critical(self, "Помилка ", "Не має даних", QMessageBox.Ok)
        else:
            numKl = np.shape(self._w)[0]
            for i in range(numKl):
                kol = 0
                for elem in self._cluster:
                    if i == elem:
                        kol = kol + 1
                self._num.append(kol)
            for i in range(self._M):
                arr = []
                for j in range(numKl):
                    arr.append(self._w[j][i]) 
                self._resW['w'+str(i)] = arr
            self._resW['Кількість'] = self._num
    
    
    def resultWeightEML (self):
        name, cl = self.clusterName()
        self._resW['Кластер'] = name
        self._resW['Кількість'] = cl
    
    
    def saveResult (self):
        if self._data.empty == True:
            QMessageBox.critical(self, "Помилка ", "Не має даних для аналзу", QMessageBox.Ok)
        elif self._norm.empty == True:
            QMessageBox.critical(self, "Помилка ", "Необхідно нормалізувати дані", QMessageBox.Ok)
        elif self._resCl.empty == True and self._resW.empty == True:
            QMessageBox.critical(self, "Помилка ", "Необхідно провести кластерний аналіз даних", QMessageBox.Ok)
        elif os.path.isfile(self._fileCluster) == True and os.path.isfile(self._fileW) == True:
            rk = pd.read_parquet(self._fileCluster)
            rw = pd.read_parquet(self._fileW)
            rk = pd.concat([rk, self._resCl], ignore_index=True)
            numKl = np.shape(self._w)[0]
            n = rw.values[:, numKl]
            for i in range(numKl):
                n[i] = n[i]+self._num[i]
            rwn = self._resW
            rwn['Кількість'] = n
            rk.to_parquet(self._fileCluster, index = False)
            rwn.to_parquet(self._fileW, index = False)
        else:
            self._resCl.to_parquet(self._fileCluster, index = False, engine='pyarrow')
            self._resW.to_parquet(self._fileW, index = False)                
            #subprocess.call(['attrib', '+h', self._fileCluster])
            #subprocess.call(['attrib', '+h', self._fileW])
    
    
    def clusterName(self):
        name = []
        clust = []
        if not self._class:
            n = self._resW.shape[0]
            j = self._resW.shape[1] - 1
            for i in range(n):
                if self._resW.values[i, j] != 0:
                    clust.append(self._resW.values[i, j])
                    name.append('Cluster ' + str(i))
        else:
            n = 0 
            k = 0
            l = 0 
            m = 0
            for i in self._class:
                if i == 0:
                    n = n +1
                elif i == 1:
                    k = k +1
                elif i == 2:
                    l = l + 1
                elif i == 3:
                    m = m +1
            arr = [n, k, l, m]
            for i in range(4) :
                if arr[i] != 0:
                    clust.append(arr[i])
                    name.append('Cluster ' + str(i))
        return name, clust
    
    
    def openResult1NNK(self):
        self.setNameFileClusterNNK1()
        self.setNameFileWeight1()
        if os.path.isfile(self._fileCluster) == True and os.path.isfile(self._fileW) == True:
            self._resCl = pd.read_parquet(self._fileCluster, engine='pyarrow')
            self._resW = pd.read_parquet(self._fileW, engine='pyarrow')
            self._num = self._resW.values[:, self._K]
            self._nv = 0
            
        else:
            QMessageBox.critical(self, "Помилка ", "Попередніх результатів не має", QMessageBox.Ok)
    
    
    def openResult2NNK(self):
        self.setNameFileClusterNNK2()
        self.setNameFileWeight2()
        if os.path.isfile(self._fileCluster) == True and os.path.isfile(self._fileW) == True:
            self._resCl = pd.read_parquet(self._fileCluster, engine='pyarrow')
            self._resW = pd.read_parquet(self._fileW, engine='pyarrow')
            self._nv = 0
            self._class = list(self._resCl.values[:,6])
        else:
            QMessageBox.critical(self, "Помилка ", "Попередніх результатів не має", QMessageBox.Ok)


    def openResultART(self):
        self.setNameFileClusterART()
        self.setNameFileWeightART()
        if os.path.isfile(self._fileCluster) == True and os.path.isfile(self._fileW) == True:
            self._resCl = pd.read_parquet(self._fileCluster)
            self._resW = pd.read_parquet(self._fileW)
            self._num = self._resW.values[:, self._K]
            self._nv = 0
        else:
            QMessageBox.critical(self, "Помилка ", "Попередніх результатів не має", QMessageBox.Ok)
    
    def openResulELM(self):
        self.setNameFileClusterELM()
        self.setNameFileWeightELM()
        if os.path.isfile(self._fileCluster) == True and os.path.isfile(self._fileW) == True:
            self._resCl = pd.read_parquet(self._fileCluster)
            self._resW = pd.read_parquet(self._fileW)
            self._num = self._resW.values[:, 1]
            self._nv = 0
        else:
            QMessageBox.critical(self, "Помилка ", "Попередніх результатів не має", QMessageBox.Ok)

    def deleteFile(self):
        self.setNameFileClusterEML()
        self.setNameFileWeightEML()
        if os.path.isfile(self._fileCluster) == True and os.path.isfile(self._fileW) == True:
            os.remove(self._fileCluster)
            os.remove(self._fileW)
        self.setNameFileClusterNNK2()
        self.setNameFileWeight2()
        if os.path.isfile(self._fileCluster) == True and os.path.isfile(self._fileW) == True:
            os.remove(self._fileCluster)
            os.remove(self._fileW)
        self.setNameFileClusterNNK1()
        self.setNameFileWeight1()
        if os.path.isfile(self._fileCluster) == True and os.path.isfile(self._fileW) == True:
            os.remove(self._fileCluster)
            os.remove(self._fileW)
        os.remove(self._fname)
        
        
        
class Password():
    def __init__(self):
        super().__init__()
        self._pass = ""
        self._filePass = ""
        
        
    def setFilePass(self):
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path     = os.path.dirname(os.path.abspath(filename))
        self._filePass = path + "\\"+ "~Pass.txt"
    
    
    def setPass(self):
        self.setFilePass()
        if os.path.isfile(self._filePass) == True:
            handle = open(self._filePass, "rb")
            pas = handle.read()
            key = b'V8N25oqUlRLRhFJDSSq2W75VS7TcbB2XASAcNnoOEzs='
            cipher = Fernet(key)
            pas = str(cipher.decrypt(pas))
            self._pass = pas[2:-1]
            handle.close()
        else:
            self._pass = "0000"
    
    def pad(self, text):
        t = text.encode('utf-8')
        return t
 
    
    def writePass(self, txt):
        key = b'V8N25oqUlRLRhFJDSSq2W75VS7TcbB2XASAcNnoOEzs='
        cipher = Fernet(key)
        pad_txt = self.pad(txt)
        encrypted_text = cipher.encrypt(pad_txt)
        handle = open(self._filePass, "wb")
        handle.write(encrypted_text)
        handle.close()
        subprocess.call(['attrib', '+h', self._filePass])
       
        
    def changePass(self, txt, txt2):
        self.setPass()
        if txt == self._pass:
            self.writePass(txt2)
        else:
            QMessageBox.critical(self, "Помилка ", "Не вірний пароль", QMessageBox.Ok)
        
    

class Ui_PassWindow(object):
    def setupUi(self, PassWindow):
        PassWindow.setObjectName("PassWindow")
        PassWindow.resize(343, 205)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        PassWindow.setPalette(palette)
        self.centralwidget = QtWidgets.QWidget(PassWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_front = QtWidgets.QLabel(self.centralwidget)
        self.label_front.setGeometry(QtCore.QRect(70, 10, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(20)
        self.label_front.setFont(font)
        self.label_front.setObjectName("label_front")
        self.pushButton_pass = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_pass.setGeometry(QtCore.QRect(110, 90, 101, 41))
        font = QtGui.QFont()
        font.setFamily("Noto Mono")
        font.setPointSize(14)
        self.pushButton_pass.setFont(font)
        self.pushButton_pass.setObjectName("pushButton_pass")
        self.text_password = QtWidgets.QTextEdit(self.centralwidget)
        self.text_password.setGeometry(QtCore.QRect(70, 50, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Symbol type A")
        self.text_password.setFont(font)
        self.text_password.setObjectName("text_password")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 140, 251, 20))
        self.label.setText("")
        self.label.setObjectName("label")
        PassWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(PassWindow)
        self.statusbar.setObjectName("statusbar")
        PassWindow.setStatusBar(self.statusbar)

        self.retranslateUi(PassWindow)
        QtCore.QMetaObject.connectSlotsByName(PassWindow)

    def retranslateUi(self, PassWindow):
        _translate = QtCore.QCoreApplication.translate
        PassWindow.setWindowTitle(_translate("PassWindow", "СlusterPro"))
        self.label_front.setText(_translate("PassWindow", "Введіть пароль"))
        self.pushButton_pass.setText(_translate("PassWindow", "OK"))



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1201, 613)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1201, 621))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.page1 = QtWidgets.QWidget()
        self.page1.setObjectName("page1")
        self.push_download = QtWidgets.QPushButton(self.page1)
        self.push_download.setGeometry(QtCore.QRect(1030, 0, 161, 31))
        self.push_download.setObjectName("push_download")
        self.push_delete_row = QtWidgets.QPushButton(self.page1)
        self.push_delete_row.setGeometry(QtCore.QRect(1030, 160, 161, 31))
        self.push_delete_row.setObjectName("push_delete_row")
        self.push_add_row = QtWidgets.QPushButton(self.page1)
        self.push_add_row.setGeometry(QtCore.QRect(1030, 130, 161, 31))
        self.push_add_row.setObjectName("push_add_row")
        self.push_delete_file = QtWidgets.QPushButton(self.page1)
        self.push_delete_file.setGeometry(QtCore.QRect(1030, 270, 161, 31))
        self.push_delete_file.setObjectName("push_delete_file")
        self.groupBox = QtWidgets.QGroupBox(self.page1)
        self.groupBox.setGeometry(QtCore.QRect(1010, 320, 181, 241))
        self.groupBox.setObjectName("groupBox")
        self.push_creat_table = QtWidgets.QPushButton(self.groupBox)
        self.push_creat_table.setGeometry(QtCore.QRect(10, 200, 171, 31))
        self.push_creat_table.setObjectName("push_creat_table")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 150, 111, 21))
        self.label.setObjectName("label")
        self.text_column = QtWidgets.QTextEdit(self.groupBox)
        self.text_column.setGeometry(QtCore.QRect(0, 80, 181, 31))
        self.text_column.setObjectName("text_column")
        self.text_file = QtWidgets.QTextEdit(self.groupBox)
        self.text_file.setGeometry(QtCore.QRect(0, 170, 181, 31))
        self.text_file.setObjectName("text_file")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 60, 121, 16))
        self.label_2.setObjectName("label_2")
        self.push_column = QtWidgets.QPushButton(self.groupBox)
        self.push_column.setGeometry(QtCore.QRect(10, 110, 171, 31))
        self.push_column.setObjectName("push_column")
        self.push_clean_table = QtWidgets.QPushButton(self.groupBox)
        self.push_clean_table.setGeometry(QtCore.QRect(10, 20, 161, 31))
        self.push_clean_table.setObjectName("push_clean_table")
        self.push_save_data = QtWidgets.QPushButton(self.page1)
        self.push_save_data.setGeometry(QtCore.QRect(1030, 190, 161, 31))
        self.push_save_data.setObjectName("push_save_data")
        self.textFileOpen = QtWidgets.QTextEdit(self.page1)
        self.textFileOpen.setGeometry(QtCore.QRect(1030, 60, 161, 31))
        self.textFileOpen.setObjectName("textFileOpen")
        self.pushOpenFile = QtWidgets.QPushButton(self.page1)
        self.pushOpenFile.setGeometry(QtCore.QRect(1030, 90, 161, 31))
        self.pushOpenFile.setObjectName("pushOpenFile")
        self.line = QtWidgets.QFrame(self.page1)
        self.line.setGeometry(QtCore.QRect(1010, 310, 181, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_5 = QtWidgets.QLabel(self.page1)
        self.label_5.setGeometry(QtCore.QRect(1040, 40, 111, 21))
        self.label_5.setObjectName("label_5")
        self.tableData = QtWidgets.QTableWidget(self.page1)
        self.tableData.setGeometry(QtCore.QRect(0, 0, 1001, 561))
        self.tableData.setObjectName("tableData")
        self.tableData.setColumnCount(0)
        self.tableData.setRowCount(0)
        self.pushCleanFile = QtWidgets.QPushButton(self.page1)
        self.pushCleanFile.setGeometry(QtCore.QRect(1030, 230, 161, 31))
        self.pushCleanFile.setObjectName("pushCleanFile")
        self.tabWidget.addTab(self.page1, "")
        self.page2 = QtWidgets.QWidget()
        self.page2.setObjectName("page2")
        self.tableResultNNK = QtWidgets.QTableView(self.page2)
        self.tableResultNNK.setGeometry(QtCore.QRect(0, 70, 871, 491))
        self.tableResultNNK.setObjectName("tableResultNNK")
        self.tableResultNNK2 = QtWidgets.QTableView(self.page2)
        self.tableResultNNK2.setGeometry(QtCore.QRect(880, 70, 301, 211))
        self.tableResultNNK2.setObjectName("tableResultNNK2")
        self.gridLayoutWidget = QtWidgets.QWidget(self.page2)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(880, 290, 301, 261))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.canvas  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout.addWidget(self.canvas)
        self.push_NNK = QtWidgets.QPushButton(self.page2)
        self.push_NNK.setGeometry(QtCore.QRect(300, 10, 131, 51))
        self.push_NNK.setObjectName("push_NNK")
        self.pushPR_NNK = QtWidgets.QPushButton(self.page2)
        self.pushPR_NNK.setGeometry(QtCore.QRect(770, 10, 201, 51))
        self.pushPR_NNK.setObjectName("pushPR_NNK")
        self.spinBox_NNK = QtWidgets.QSpinBox(self.page2)
        self.spinBox_NNK.setGeometry(QtCore.QRect(160, 30, 42, 22))
        self.spinBox_NNK.setMinimum(2)
        self.spinBox_NNK.setObjectName("spinBox_NNK")
        self.label_3 = QtWidgets.QLabel(self.page2)
        self.label_3.setGeometry(QtCore.QRect(150, 10, 151, 16))
        self.label_3.setObjectName("label_3")
        self.push_NNK_EM = QtWidgets.QPushButton(self.page2)
        self.push_NNK_EM.setGeometry(QtCore.QRect(440, 10, 181, 51))
        self.push_NNK_EM.setObjectName("push_NNK_EM")
        self.push_normNNK = QtWidgets.QPushButton(self.page2)
        self.push_normNNK.setGeometry(QtCore.QRect(10, 12, 131, 51))
        self.push_normNNK.setObjectName("push_normNNK")
        self.push_seveNNK = QtWidgets.QPushButton(self.page2)
        self.push_seveNNK.setGeometry(QtCore.QRect(634, 10, 131, 51))
        self.push_seveNNK.setObjectName("push_seveNNK")
        self.pushPR_NNK2 = QtWidgets.QPushButton(self.page2)
        self.pushPR_NNK2.setGeometry(QtCore.QRect(980, 10, 201, 51))
        self.pushPR_NNK2.setObjectName("pushPR_NNK2")
        self.tabWidget.addTab(self.page2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tableResultNNAR = QtWidgets.QTableView(self.tab)
        self.tableResultNNAR.setGeometry(QtCore.QRect(0, 70, 871, 491))
        self.tableResultNNAR.setObjectName("tableResultNNAR")
        self.tableResultNNAR2 = QtWidgets.QTableView(self.tab)
        self.tableResultNNAR2.setGeometry(QtCore.QRect(880, 70, 301, 211))
        self.tableResultNNAR2.setObjectName("tableResultNNAR2")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.tab)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(880, 290, 301, 261))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.canvas2  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout_2.addWidget(self.canvas2)
        self.push_NNAR = QtWidgets.QPushButton(self.tab)
        self.push_NNAR.setGeometry(QtCore.QRect(190, 10, 181, 51))
        self.push_NNAR.setObjectName("push_NNAR")
        self.pushPR_NNAR = QtWidgets.QPushButton(self.tab)
        self.pushPR_NNAR.setGeometry(QtCore.QRect(750, 10, 181, 51))
        self.pushPR_NNAR.setObjectName("pushPR_NNAR")
        self.push_normAR = QtWidgets.QPushButton(self.tab)
        self.push_normAR.setGeometry(QtCore.QRect(10, 10, 171, 51))
        self.push_normAR.setObjectName("push_normAR")
        self.push_saveNNAR = QtWidgets.QPushButton(self.tab)
        self.push_saveNNAR.setGeometry(QtCore.QRect(570, 10, 171, 51))
        self.push_saveNNAR.setObjectName("push_saveNNAR")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tableResultEML = QtWidgets.QTableView(self.tab_2)
        self.tableResultEML.setGeometry(QtCore.QRect(0, 70, 871, 491))
        self.tableResultEML.setObjectName("tableResultEML")
        self.tableResultELM2 = QtWidgets.QTableView(self.tab_2)
        self.tableResultELM2.setGeometry(QtCore.QRect(880, 70, 301, 211))
        self.tableResultELM2.setObjectName("tableResultELM2")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.tab_2)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(880, 290, 301, 261))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.canvas3  = MplCanvas(self, width=5, height=4, dpi=100)
        self.gridLayout_3.addWidget(self.canvas3)
        self.push_EML_EM = QtWidgets.QPushButton(self.tab_2)
        self.push_EML_EM.setGeometry(QtCore.QRect(150, 10, 181, 51))
        self.push_EML_EM.setObjectName("push_EML_EM")
        self.push_normELM = QtWidgets.QPushButton(self.tab_2)
        self.push_normELM.setGeometry(QtCore.QRect(10, 10, 131, 51))
        self.push_normELM.setObjectName("push_normELM")
        self.pushButton = QtWidgets.QPushButton(self.tab_2)
        self.pushButton.setGeometry(QtCore.QRect(340, 10, 171, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushPR_EML2 = QtWidgets.QPushButton(self.tab_2)
        self.pushPR_EML2.setGeometry(QtCore.QRect(520, 10, 181, 51))
        self.pushPR_EML2.setObjectName("pushPR_EML2")
        self.tabWidget.addTab(self.tab_2, "")
        self.page3 = QtWidgets.QWidget()
        self.page3.setObjectName("page3")
        self.label_4 = QtWidgets.QLabel(self.page3)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 421, 251))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_4.setFont(font)
        self.label_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_4.setObjectName("label_4")
        self.push_change_pass = QtWidgets.QPushButton(self.page3)
        self.push_change_pass.setGeometry(QtCore.QRect(1000, 260, 131, 31))
        self.push_change_pass.setObjectName("push_change_pass")
        self.label_7 = QtWidgets.QLabel(self.page3)
        self.label_7.setGeometry(QtCore.QRect(830, 10, 321, 231))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.page3)
        self.label_8.setGeometry(QtCore.QRect(440, 0, 481, 311))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.page3)
        self.label_9.setGeometry(QtCore.QRect(10, 290, 671, 261))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.page3)
        self.label_10.setGeometry(QtCore.QRect(710, 420, 441, 141))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_6 = QtWidgets.QLabel(self.page3)
        self.label_6.setGeometry(QtCore.QRect(710, 310, 431, 111))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.tabWidget.addTab(self.page3, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ClusterPro"))
        self.push_download.setText(_translate("MainWindow", "Завантажити"))
        self.push_delete_row.setText(_translate("MainWindow", "Видалити строку"))
        self.push_add_row.setText(_translate("MainWindow", "Добавити строку"))
        self.push_delete_file.setText(_translate("MainWindow", "Видалити файл"))
        self.groupBox.setTitle(_translate("MainWindow", "Створення таблиці"))
        self.push_creat_table.setText(_translate("MainWindow", "Створити таблицю"))
        self.label.setText(_translate("MainWindow", "Ім\'я файлу"))
        self.label_2.setText(_translate("MainWindow", "Ім\'я стовпчика"))
        self.push_column.setText(_translate("MainWindow", "Добавити до таблиці"))
        self.push_clean_table.setText(_translate("MainWindow", "Очистити таблицю"))
        self.push_save_data.setText(_translate("MainWindow", "Зберегти дані"))
        self.pushOpenFile.setText(_translate("MainWindow", "Відкрити файл"))
        self.label_5.setText(_translate("MainWindow", "Ім\'я файлу"))
        self.pushCleanFile.setText(_translate("MainWindow", "Очистити файл"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.page1), _translate("MainWindow", "Дані"))
        self.push_NNK.setText(_translate("MainWindow", "Провести\n"
" аналіз"))
        self.pushPR_NNK.setText(_translate("MainWindow", "Попередні результати"))
        self.label_3.setText(_translate("MainWindow", "Кількість кластерів"))
        self.push_NNK_EM.setText(_translate("MainWindow", "Кластерний аналіз \n"
"електрообладнання"))
        self.push_normNNK.setText(_translate("MainWindow", "Нормалізувати\n"
" дані"))
        self.push_seveNNK.setText(_translate("MainWindow", "Зберігти \n"
"дані"))
        self.pushPR_NNK2.setText(_translate("MainWindow", "Попередній аналіз \n"
"електрообладнання"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.page2), _translate("MainWindow", "НМ Кохонена"))
        self.push_NNAR.setText(_translate("MainWindow", "Провести аналіз"))
        self.pushPR_NNAR.setText(_translate("MainWindow", "Попередні результати"))
        self.push_normAR.setText(_translate("MainWindow", "Нормалізувати дані"))
        self.push_saveNNAR.setText(_translate("MainWindow", "Зберігти дані"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "НМАР"))
        self.push_EML_EM.setText(_translate("MainWindow", "Кластерний аналіз \n"
"електрооблааднання"))
        self.push_normELM.setText(_translate("MainWindow", "Нормалізувати\n"
" дані"))
        self.pushButton.setText(_translate("MainWindow", "Зберігти \n"
"дані"))
        self.pushPR_EML2.setText(_translate("MainWindow", "Попередній аналіз\n"
"електрообладнання"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "ЕМН"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Можливості програми:</span></p><p>- завнтаження файлів з табличними даними (.excel, .txt, .parquet)</p><p>- редагування даних</p><p>- створення нової таблиці і файлу</p><p>- нормалізація даних</p><p>- кластерний аналіз трьох типів</p><p>- збереження результату аналізу</p><p>- відображення попередніх результатів кластеризації</p><p>- видалення файлу</p><p>- очищення від даних файлу</p></body></html>"))
        self.push_change_pass.setText(_translate("MainWindow", "Змінити пороль"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; text-decoration: underline;\">Алгоритм кластерного аналізу:</span></p><p><span style=\" text-decoration: underline;\">1) Завантажити данні або створити </span></p><p><span style=\" text-decoration: underline;\">2) При створенні обов\'язково збергти</span></p><p><span style=\" text-decoration: underline;\">3) Нормалізувати дані за необхідністю</span></p><p><span style=\" text-decoration: underline;\">4) Провести кластерний аналіз обравши загальний чи </span></p><p><span style=\" text-decoration: underline;\">електрообладнання</span></p><p><span style=\" text-decoration: underline;\">5) Зберегти результат за власним рішенням</span></p><p><span style=\" text-decoration: underline;\">6) При завантаженні попередніх результатів необхідно</span></p><p><span style=\" text-decoration: underline;\">відкрити файл з відповідними даними</span></p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; text-decoration: underline;\">Створення таблиці:</span></p><p><span style=\" text-decoration: underline;\">1) Необхідно очистити таблицю, якщо в ній є дані</span></p><p><span style=\" text-decoration: underline;\">2) Почерзі добавляти назви стовпчиків</span></p><p><span style=\" text-decoration: underline;\">3) Вказати ім\'я файлу без шляхів і розширення. </span></p><p><span style=\" text-decoration: underline;\">Ім\'я файлу повинно відображати загальний зміст даних.</span></p><p><span style=\" text-decoration: underline;\">Наприклад назва обладнання, яке підлягає аналізу</span></p><p><span style=\" text-decoration: underline;\">4) Натиснути створити таблицю</span></p><p><span style=\" text-decoration: underline;\">5) Добавити строки та заповнити їх</span></p><p><span style=\" text-decoration: underline;\">6) За необхідністю видалити строку з початку натиснути</span></p><p><span style=\" text-decoration: underline;\">на необхідну строку потім на кнопку Видалити строку</span></p><p><span style=\" text-decoration: underline;\">7 )Зберегти дані</span></p><p><span style=\" text-decoration: underline;\">8) За необхідністю можна очистити таблицю у файлі, або повністю видалити файл</span></p></body></html>"))
        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">Зауваження!</span></p><p><span style=\" font-size:9pt; font-weight:600;\">1) Таблиця заповнюється в усіх стовбцях і з відповідним типом</span></p><p><span style=\" font-size:9pt; font-weight:600;\">2) У разі видалення таблиці видаляються усі результати кластерізації</span></p><p><span style=\" font-size:9pt; font-weight:600;\">3) При нормалізації, кількість даних повино бути більше 2</span></p><p><span style=\" font-size:9pt; font-weight:600;\">4) Один вид кластеризації (наприклад НМФР - аналіз електрообладнання) проводиться один раз, </span></p><p><span style=\" font-size:9pt; font-weight:600;\">тобто дані, що вже проходили кластеризацію більше не аналізуються</span></p><p><span style=\" font-size:9pt; font-weight:600;\">5) Збережені дані у файлу не можливо відредагувати, їх можно тільки повністю видалити </span></p><p><span style=\" font-size:9pt; font-weight:600;\">6) Кнопка </span><span style=\" font-size:9pt; font-weight:600; font-style:italic;\">Зберегти дані</span><span style=\" font-size:9pt; font-weight:600;\"> зберігає усі дані у відображеній таблиці у файлі .parquet, який видно тільки у</span></p><p><span style=\" font-size:9pt; font-weight:600;\"> програмі з тим же ім\'ям як у початкового файлу. Тому надалі, якщо данні будуть доповнюватися у</span></p><p><span style=\" font-size:9pt; font-weight:600;\"> програмі треба відкривати файл написавши його назву у текстоваму полі.</span></p></body></html>"))
        self.label_10.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-style:italic;\">Пояснення до класторного аналізу елоктрообладнання:</span></p><p><span style=\" font-style:italic;\">klaster 0 - обладнання працює нормально</span></p><p><span style=\" font-style:italic;\">klaster 1 - потрібен поточний ремонт</span></p><p><span style=\" font-style:italic;\">klaster 2 - потрібен капітальний ремонт</span></p><p><span style=\" font-style:italic;\">klaster 3 - обладнання не підлягає ремонту</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; font-style:italic;\">Види кластерного аналізу:</span></p><p><span style=\" font-weight:600; font-style:italic;\">НМ Кохонена - нейронна мережа Кохонена</span></p><p><span style=\" font-weight:600; font-style:italic;\">НМАР - нейронна мережа адаптивного резонансу ART2</span></p><p><span style=\" font-weight:600; font-style:italic;\">ЕМН - екстремальне машинне навчання<br/></span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.page3), _translate("MainWindow", "Про програму"))



class Ui_ChangeWindow(object):
    def setupUi(self, ChangeWindow):
        ChangeWindow.setObjectName("ChangeWindow")
        ChangeWindow.resize(425, 190)
        font = QtGui.QFont()
        font.setPointSize(12)
        ChangeWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(ChangeWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.text_oldPass = QtWidgets.QTextEdit(self.centralwidget)
        self.text_oldPass.setGeometry(QtCore.QRect(210, 30, 191, 31))
        self.text_oldPass.setObjectName("text_oldPass")
        self.text_newPass = QtWidgets.QTextEdit(self.centralwidget)
        self.text_newPass.setGeometry(QtCore.QRect(210, 70, 191, 31))
        self.text_newPass.setObjectName("text_newPass")
        self.push_changePass = QtWidgets.QPushButton(self.centralwidget)
        self.push_changePass.setGeometry(QtCore.QRect(150, 110, 121, 31))
        self.push_changePass.setObjectName("push_changePass")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 30, 171, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 70, 171, 21))
        self.label_2.setObjectName("label_2")
        ChangeWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(ChangeWindow)
        self.statusbar.setObjectName("statusbar")
        ChangeWindow.setStatusBar(self.statusbar)

        self.retranslateUi(ChangeWindow)
        QtCore.QMetaObject.connectSlotsByName(ChangeWindow)

    def retranslateUi(self, ChangeWindow):
        _translate = QtCore.QCoreApplication.translate
        ChangeWindow.setWindowTitle(_translate("ChangeWindow", "Зміна паралю"))
        self.push_changePass.setText(_translate("ChangeWindow", "Змінити пароль"))
        self.label.setText(_translate("ChangeWindow", "<html><head/><body><p>Введіть старий пароль</p></body></html>"))
        self.label_2.setText(_translate("ChangeWindow", "<html><head/><body><p>Введіть новий пароль</p></body></html>"))


class PresenterChangePassword(QMainWindow, Ui_ChangeWindow):
    def __init__(self):
        super(PresenterChangePassword, self).__init__()
        self.setupUi(self)
        self.obj = Password()
        self.push_changePass.pressed.connect(self.changeP)
        
    def changeP(self):
        txt = self.text_oldPass.toPlainText()
        txt2 = self.text_newPass.toPlainText()
        self.obj.changePass(txt, txt2)
        


class PresenterMain(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super(PresenterMain, self).__init__()
        self.N = 0
        self.setupUi(self)
        self.obj = Result() 
        self.newData = pd.DataFrame()
        self.push_download.pressed.connect(self.downloadF)
        self.pushOpenFile.pressed.connect(self.openP)
        self.tableData.setRowCount(self.N)
        self.push_add_row.pressed.connect(self.addRow)
        self.push_save_data.pressed.connect(self.saveTable)
        self.push_delete_row.pressed.connect(self.deleteRowNewData)
        self.pushCleanFile.pressed.connect(self.cleanData)
        self.push_clean_table.pressed.connect(self.cleanTable)
        self.push_column.pressed.connect(self.addNewHeaders)
        self.push_creat_table.pressed.connect(self.creatFile)
        self.push_NNK.pressed.connect(lambda k=1: self.NNK(k))
        self.push_normNNK.pressed.connect(lambda k=1:self.norm(k))
        self.push_NNK_EM.pressed.connect(lambda k=2: self.NNK(k))
        self.push_seveNNK.pressed.connect(self.obj.saveResult)
        self.push_normAR.pressed.connect(lambda k=2: self.norm(k))
        self.push_NNAR.pressed.connect(self.ART2)
        self.push_EML_EM.pressed.connect(self.EML)
        self.pushPR_NNAR.pressed.connect(self.prevResultART)
        self.push_saveNNAR.pressed.connect(self.obj.saveResult)
        self.pushButton.pressed.connect(self.obj.saveResult)
        self.push_normELM.pressed.connect(lambda k=3:self.norm(k))
        self.push_change_pass.pressed.connect(self.check)
        self.pushPR_NNK.pressed.connect(lambda k=1:self.prevResultNNK(k))
        self.pushPR_NNK2.pressed.connect(lambda k=2:self.prevResultNNK(k))
        self.push_delete_file.pressed.connect(self.deleteF)
    
    
    def deleteF(self):
        self.cleanTable()
        self.obj.deleteFile()
        
    
    
    def prevResultNNK (self, k):
        if k == 1:
            self.obj.openResult1NNK()
        else:
            self.obj.openResult2NNK()
        model1 = pandasModel(self.obj._resCl)
        self.tableResultNNK.setModel(model1)
        model2 = pandasModel(self.obj._resW)
        self.tableResultNNK2.setModel(model2)
        name  = []
        name, clust = self.obj.clusterName()
        self.canvas2.axes.clear()
        self.canvas.axes.pie(clust, labels = name)
        self.canvas.draw()
        
        
    def prevResultART (self):
        self.obj.openResultART()   
        model1 = pandasModel(self.obj._resCl)
        self.tableResultNNAR.setModel(model1)
        model2 = pandasModel(self.obj._resW)
        self.tableResultNNAR2.setModel(model2)
        name  = []
        name, clust = self.obj.clusterName()
        self.canvas2.axes.clear()
        self.canvas2.axes.pie(clust, labels = name)
        self.canvas2.draw()
        
    
    def EML(self):
        if self.obj._data.empty == True:
            QMessageBox.critical(self, "Помилка ", "Немає даних", QMessageBox.Ok)
        elif self.obj._norm.empty == True:
            self.obj.dataNorm()
        else:
            self.obj._resCl = pd.DataFrame()
            self.obj._resW = pd.DataFrame()
            self.obj._num = []
            self.obj._cluster = []
            self.obj._w = []
            self.obj._class = []
            self.obj.to_analyzeEML()
            self.obj.resultClustering()
            self.obj.resultWeightEML()
            model1 = pandasModel(self.obj._resCl)
            self.tableResultEML.setModel(model1)
            model2 = pandasModel(self.obj._resW)
            self.tableResultELM2.setModel(model2)
            name  = []
            name, clust = self.obj.clusterName()
            self.canvas3.axes.clear()
            self.canvas3.axes.pie(clust, labels = name)
            self.canvas3.draw()
    
    
    def ART2(self):
        if self.obj._data.empty == True:
            QMessageBox.critical(self, "Помилка ", "Немає даних", QMessageBox.Ok)
        elif self.obj._norm.empty == True:
            self.obj.dataNorm()
        else:
            self.obj._K = 0
            self.obj._nv = 0
            self.obj._resCl = pd.DataFrame()
            self.obj._resW = pd.DataFrame()
            self.obj._num = []
            self.obj._cluster = []
            self.obj._w = []
            self.obj._class = []
            self.obj.to_analyzeART()
            self.obj.resultClustering()
            self.obj.resultWeight()
            model1 = pandasModel(self.obj._resCl)
            self.tableResultNNAR.setModel(model1)
            model2 = pandasModel(self.obj._resW)
            self.tableResultNNAR2.setModel(model2)
            name  = []
            name, clust = self.obj.clusterName()
            self.canvas2.axes.clear()
            self.canvas2.axes.pie(clust, labels = name)
            self.canvas2.draw()
      
    
    def  NNK(self, k):
        if self.obj._data.empty == True:
            QMessageBox.critical(self, "Помилка ", "Немає даних", QMessageBox.Ok)
        elif self.obj._norm.empty == True:
            self.obj.dataNorm()
        else:
            self.obj._K = 0
            self.obj._nv = 0
            self.obj._resCl = pd.DataFrame()
            self.obj._resW = pd.DataFrame()
            self.obj._num = []
            self.obj._cluster = []
            self.obj._w = []
            self.obj._class = []
            if k == 1:
                c = self.spinBox_NNK.value()
                self.obj.to_analyzeNNK(k, c)
            else:
                self.obj.to_analyzeNNK(k)
            self.obj.resultClustering()
            self.obj.resultWeight()
            model1 = pandasModel(self.obj._resCl)
            self.tableResultNNK.setModel(model1)
            model2 = pandasModel(self.obj._resW)
            self.tableResultNNK2.setModel(model2)
            name  = []
            name, clust = self.obj.clusterName()
            self.canvas.axes.clear()
            self.canvas.axes.pie(clust, labels = name)
            self.canvas.draw()
    
    def norm(self, k):
        self.obj.transform()
        model = pandasModel(self.obj._norm)
        if k == 1:
            self.tableResultNNK.setModel(model)
        elif k == 2:
            self.tableResultNNAR.setModel(model)
        elif k == 3:
            self.tableResultEML.setModel(model)
    
    
    def creatFile(self):
        fn = self.text_file.toPlainText()
        self.obj.creatFileData(fn)
        self.text_file.clear()
    
    
    def addNewHeaders(self):
        head = self.text_column.toPlainText()
        self.obj._data[head] = []
        self.showTab(self.obj._data, self.obj._M)
        self.tableData.show()
        self.text_column.clear()
    
    
    def cleanTable(self):
        self.tableData.clear
        self.N = 0
        self.tableData.setRowCount(self.N)
        self.tableData.setColumnCount(0)
        self.tableData.show()
        self.obj._M = 0
        self.obj._N = 0
        self.obj._data = pd.DataFrame()
        
        
    def cleanData(self):
        self.obj.cleanFile()
        self.tableData.clear()
        self.showTab(self.obj._data,self.obj._M)
        self.N = 0
        self.tableData.setRowCount(self.N)
        self.tableData.show()
        
        
    def deleteRowNewData(self):
        row = self.tableData.currentRow()
        column = self.tableData.currentColumn()
        if row+1 > self.obj._N:
            if self.tableData.item(row, column) != None:
                df = self.getNewData()
                if np.shape(df)[0] > 1:
                    self.newData = self.obj.convertNewData(df)
                    self.newData.drop([row-self.obj._N], inplace=True)
                    self.showTab(self.newData, self.newData.shape[1])
            self.N = self.N-1
            self.tableData.setRowCount(self.N)        
            self.tableData.show()
        else:
            QMessageBox.critical(self, "Помилка ", "Не можна видаляти цю строку", QMessageBox.Ok)
   
    
    def getNewData(self):
        n = self.obj._N
        rows = self.tableData.rowCount() - n
        
        columns = self.tableData.columnCount()

        df = [[0] * columns for i in range(rows)]
        try:
            for i in range(rows):            
                for j in range(columns):                
                    df[i][j] = self.tableData.item(i+n, j).text() 
        except:
                QMessageBox.critical(self, "Помилка ", "Не повністю заповнена строка", QMessageBox.Ok)
        
        return df
    
    
    def saveTable(self):
        df = self.getNewData()        
        self.newData = self.obj.convertNewData(df)
        if self.newData.empty == False:
            self.obj.saveNewData(self.newData)
        
    
    def addRow (self):
        self.N = self.N+1
        self.tableData.setRowCount(self.N)
        self.tableData.setItem(self.N, 0, QTableWidgetItem(str(self.N)))
        self.tableData.show()
    
        
    def showTab (self, dt, m):
        if dt.empty == False:
            data = pd.DataFrame()
            liss = list(dt)
            for i in range (m):
                data[liss[i]] = dt.values[:,i]
            headers = dt.columns.values.tolist()
            self.tableData.setColumnCount(len(headers))
            self.tableData.setHorizontalHeaderLabels(headers)
            for i, row in data.iterrows():
                # Добавление строки
                self.tableData.setRowCount(self.tableData.rowCount() + 1)
  
                for j in range(self.tableData.columnCount()):
                    self.tableData.setItem(i, j, QTableWidgetItem(str(row[j])))
        else:
            headers = dt.columns.values.tolist()
            self.tableData.setColumnCount(len(headers))
            self.tableData.setHorizontalHeaderLabels(headers)
        
             
    
    def downloadF (self):
        self.obj.showDialog()
        self.N = self.obj._N
        self.showTab(self.obj._data, self.obj._M)
        self.tableData.show()
        
        
    def openP(self):
        file = self.textFileOpen.toPlainText()
        self.obj.openParquet(file)
        self.showTab(self.obj._data, self.obj._M)
    
    
    def check(self):
        self.changeWindow = PresenterChangePassword()
        self.changeWindow.show()
        
        
        
class PresenterPassword(QMainWindow, Ui_PassWindow):
    def __init__(self):
        super(PresenterPassword, self).__init__()
        self.setupUi(self)
        self.obj = Password()
        self.pushButton_pass.clicked.connect(self.check)
        
        
    def check(self):
        self.obj.setPass()
        txt = self.text_password.toPlainText()
        if self.obj._pass == txt:
            self.close()
            self.mainWindow = PresenterMain()
            self.mainWindow.show()
        else:
            QMessageBox.critical(self, "Помилка ", "Не вірний пароль", QMessageBox.Ok)
  
      
def main():
    app = QApplication(sys.argv)
    #appctxt = ApplicationContext()
    window = PresenterPassword()
    window.show()
    sys.exit(app.exec_())
    #sys.exit(appctxt.app.exec_())

if __name__ == "__main__":
    main()

 
    
