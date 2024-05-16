# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 00:38:14 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
def SAT ():
    plt.rcParams['font.size'] = 18
    filename = 'cheetah.png' 
    test_im = io.imread(filename)
    rows = test_im.shape[0] # кількість рядків
    clms = test_im.shape[1] # кількість колонок
    image_gray = np.zeros ((rows , clms,1), dtype=np.uint8)
    for i in  range (rows):
        for j in  range (clms):
            image_gray [i, j, 0] = 0.299*test_im [i, j, 0]+0.587*test_im [i, j, 1]+0.114*test_im [ i, j, 2]
    im = []
    i = 0
    # считаем сумму интенсивностей
    while i < rows:
        mt = []
        j = 0 
        while j < clms:
            s = 0
            for f in range (i):
                for t in  range (j):
                    s = s + image_gray [f, t, 0]       
            mt.append(s)
            j=j+1
        im.append(mt)
        i= i+1
    # Показатель Хаара
    h = 30
    h2 = int(h/2) 
    leng = 30
    minI = rows*clms*170
    mini = rows
    minj = clms
    for i in range(rows-h):
        for j in range(clms-leng):
            Iw = 0
            Ib = 0
            if i == 0:
                Ib = im[i+h][j+leng]- im[i+h][j]
                Iw = im[i+h2][j+leng]- im[i+h2][j]
            elif j == 0:
                Ib = im[i+h][j+leng]- im[i][j+leng]
                Iw = im[i+h2][j+leng]- im[i][j+leng]
            else:
                Ib = im[i+h][j+leng]+im[i][j]-im[i][j+leng]-im[i+h][j]
                Iw = im[i+h2][j+leng]+im[i][j]-im[i][j+leng]-im[i+h2][j]
            H = Ib-Iw
            if minI > H:
                minI = H
                mini = i
                minj = j
    print(mini, minj, minI) # верхняя левая точка и показатель Хаара
    #Рисуем область которая нашлась
    for i in range(mini, mini+h):
        for j in range(minj, minj+leng):
            if i < h2+mini:
                image_gray[i,j,0] = 255
            else:
                image_gray[i,j,0] = 1
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax = axes.ravel()
    ax[0].imshow(test_im)
    ax[0].set_title("ORIGINAL IMAGE")
    ax[1].imshow(image_gray,  cmap="gray")
    ax[1].set_title("IMAGE")
    plt.show() 
SAT()
    