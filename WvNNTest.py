# -*- coding: utf-8 -*-
"""
Created on Fri May 13 23:17:39 2022

@author: AbcdRa
"""

from WvNN import WvLayer
import cv2
import numpy as np

test_path = "D:/Projects/Python/Wavelets/dataset/test.bmp"
Wv = WvLayer.load("stc2.wv")
im = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
x = im.reshape(1600)/1000
y = Wv.get_Y(x)
max_y = np.argmax(y)
c = ["квадрат", "треугольник","круг"]
print(y )
print(c[max_y])

