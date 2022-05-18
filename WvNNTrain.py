# -*- coding: utf-8 -*-
"""
Created on Fri May 13 22:10:37 2022

@author: AbcdRa
"""

import cv2
import numpy as np
from NWNN import WNN

dataset_path = "D:/Projects/Python/Wavelets/dataset/"

ext = ".bmp"
N = 27
sqrs = [ cv2.imread(dataset_path+f"s{i}"+ext, cv2.IMREAD_GRAYSCALE) for i in range(1,N)]
trs = [ cv2.imread(dataset_path+f"t{i}"+ext, cv2.IMREAD_GRAYSCALE) for i in range(1,N)]
crcs = [cv2.imread(dataset_path+f"c{i}"+ext, cv2.IMREAD_GRAYSCALE) for i in range(1,N)]

X = []
Y = []
for sqr in sqrs:
    X.append(sqr.reshape(1600)/1600)
    Y.append(np.array([1.0,0,0]))
for trs in trs:
    X.append(trs.reshape(1600)/1600)
    Y.append(np.array([0,1.0,0]))
for crc in crcs:
    X.append(crc.reshape(1600)/1600)
    Y.append(np.array([0,0,1.0]))
    
W = WNN(1600,3, 2, 1, 100)
#Wv = WvLayer.load("stc2.wv")
#W = WNN.load("stcN.wv")
X = np.array(X)
Y = np.array(Y)
#Wv = WvLayer(1600, 3, 80)
#Wv = WvLayer.load("stc.wv")
for i in range(10):
    W.fit(X, Y, 500, 0.4, 0.004)
    predict = W.predict(X)
    print(predict*2-1)
    print(sum(np.sum((predict - Y)**2, axis=1)))

W.save("stcN.wv")

