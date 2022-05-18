# -*- coding: utf-8 -*-
"""
Created on Fri May 13 23:17:39 2022

@author: AbcdRa
"""

from NWNN import WNN
import cv2
import numpy as np

test_path = "D:/Projects/Python/Wavelets/dataset2/test.bmp"
W = WNN.load("stcN.wv")
im = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
x = im.reshape(100)/2550
y = W.get_y(x)
max_y = np.argmax(y)
c = ["горизонтальная", "вертикальная" ]
print(y )
print(c[max_y])

