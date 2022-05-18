# -*- coding: utf-8 -*-
"""
Created on Fri May 13 22:10:37 2022

@author: AbcdRa
"""

import cv2
import numpy as np
from NWNN import WNN

dataset_path = "D:/Projects/Python/Wavelets/dataset2/"

ext = ".bmp"
N = 12
hs = [ cv2.imread(dataset_path+f"h{i}"+ext, cv2.IMREAD_GRAYSCALE) for i in range(1,N)]
vs = [ cv2.imread(dataset_path+f"v{i}"+ext, cv2.IMREAD_GRAYSCALE) for i in range(1,N)]

X = []
Y = []
for h in hs:
    X.append(h.reshape(100)/2550)
    Y.append(np.array([1.0,0]))
for v in vs:
    X.append(v.reshape(100)/2550)
    Y.append(np.array([0,1.0]))

    
#W = WNN(100,2, 3, 1, 50)
#Wv = WvLayer.load("stc2.wv")
W = WNN.load("stcN.wv")
X = np.array(X)
Y = np.array(Y)
#Wv = WvLayer(1600, 3, 80)
#Wv = WvLayer.load("stc.wv")
for i in range(10):
    W.fit(X, Y, 1000, 0.5, 0.1)
    predict = W.predict(X)
    print(predict*2-1)
    print(sum(np.sum((predict - Y)**2, axis=1)))

W.save("stcN.wv")

