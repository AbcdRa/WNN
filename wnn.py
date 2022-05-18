# -*- coding: utf-8 -*-
"""
Created on Wed May 11 21:30:24 2022

@author: AbcdRa
"""

#https://core.ac.uk/download/pdf/10637269.pdf

import numpy as np
import matplotlib.pyplot as plt

def mother_wv(x):
    return (1-x**2)*np.exp(-x**2/2)


#Производная от материнского вейвлета
def mother_wv_der(x):
    return (x**3-3*x) * np.exp(-x**2/2)


#w1 - [wu, ws]
#wu - веса сдвигов
#ws - веса сжатия
#l - кол-во HU
def init_w1(X, l):
    #Разброс каждого из входов
    Ns = np.min(X, axis=0)
    Ms = np.max(X, axis=0)
    #Кол-во входов
    m = X.shape[1] 
    wu = (np.ones((m, l)).T * 0.5* (Ms+Ns)).T
    ws = (np.ones((m, l)).T * 0.2* (Ms-Ns)).T
    return np.array([wu, ws])

#
def init_w0(X):
    return np.random.rand(X.shape[1])

#l - кол-во HU
def init_w2(l):
    return np.random.rand(l+1)
    

def multidim_wv(x, j, w1):
    wu, ws = w1
    #Кол-во входов
    m = x.shape[0]
    #Результат
    r = 1
    
    for i in range(m):
        z_ij = (x[i] - wu[i][j])/ws[i][j]
        r *= mother_wv(z_ij)
    return r



def init_w(X, l):
    m = X.shape[0]
    w0 = init_w0(X)
    w1 = init_w1(X, l)
    w2 = init_w2(l)
    return w0, w1, w2


D = []


def g(x, w):
    w0, w1, w2 = w
    m = x.shape[0]
    l = w2.shape[0] - 1
    #Константа
    c = w2[-1]
    s1 = 0
    for j in range(l):
        s1 += w2[j]*multidim_wv(x, j, w1)
    s2 = 0
    for i in range(m):
        s2 += w0[i]*x[i]
    global D
    D = [s1,s2, c]
    return s1 + s2 + c


def get_part_der_bias(w, x):
    return 1

def get_part_der_w0(w, x):
    return x


def get_part_der_w1(w, x):
    w0, w1, w2 = w
    wu, ws = w1
    m = w0.shape[0];
    l = w2.shape[0]-1
    der_wu = np.ones(wu.shape)
    der_ws = np.ones(wu.shape)
    for j in range(l):
        for i in range(m):
            r = 1
            for k in range(m):
                z_kj = (x[k] - wu[k][j])/ws[k][j]
                if k != i: 
                    r *= mother_wv(z_kj)
            z_ij = (x[i] - wu[i][j])/ws[i][j]
            der_wu[i][j] = r * -w2[j]/ws[i][j] * mother_wv_der(z_ij)
            der_ws[i][j] = der_wu[i][j]*z_ij
    return der_wu, der_ws
    

def get_part_der_w2(w, x):
    w0, w1, w2 = w
    l = w2.shape[0]-1
    der = np.ones(l+1)
    for j in range(l):
        der[j] = multidim_wv(x, j, w1)
    global D
    D = der
    return der

def get_ep(y, target_y):
    return target_y - y
    
    
def back_propagation(x, target_y, w, prev_w, learn_rate=0.4, momentum=0.1):
    prev_w0, prev_w1, prev_w2 = prev_w
    prev_wu, prev_ws = prev_w1
    w0, w1, w2 = w
    wu, ws = w1
    y = g(x, w)
    ep = get_ep(y, target_y)
    print(ep)
    der_wu, der_ws = get_part_der_w1(w, x)
    new_w0 = w0 + learn_rate*ep*get_part_der_w0(w, x) + momentum*(w0-prev_w0)

    
    new_wu = wu + learn_rate*ep*der_wu + momentum*(wu-prev_wu)
    new_ws = ws + learn_rate*ep*der_ws + momentum*(ws-prev_ws)
    new_w1 = (new_wu, new_ws)
    new_w2 = w2 + learn_rate*ep*get_part_der_w2(w, x) + momentum*(w2-prev_w2)

    return new_w0, new_w1, new_w2

def learn(X, Y, w, iter_num=10):
    prev_w = w[0].copy(), w[1].copy(), w[2].copy()
    for i in range(iter_num):
        for j in range(len(X)):
            new_w = back_propagation(X[j], Y[j][0], w, prev_w,learn_rate=-0.3)
            prev_w = w
            w = new_w
    return w



    
# X = np.array([[0,1],[0,0],[1,0],[1,1]])
# Y = np.array([[1],[0],[1],[0]])
X = np.linspace(-6,6, 3)
Y = np.cos(X)+np.random.rand(3)
X = X.reshape((3,1))
Y = Y.reshape((3,1))

w = init_w(X, 5)
# w0 = np.array([0.07293035, 0.32613689])
# w2 = np.array([0.66282553, 0.60149946, 0.40573423, 0.02097707, 0.1486961 ])
# wu = np.array([[0.5, 0.5, 0.5, 0.5],
#  [0.5, 0.5, 0.5, 0.5]])
# ws = np.array([[0.2,0.2, 0.2, 0.2],
#  [0.2, 0.2, 0.2, 0.2]])
# w1 = np.array([wu, ws])
# w = w0, w1, w2
prev_w = w[0].copy(), w[1].copy(), w[2].copy()


learn(X,Y, w)
#w = learn(X, Y, w, 100)

#itog = g(X[0], w), g(X[1], w), g(X[2], w), g(X[3], w)


