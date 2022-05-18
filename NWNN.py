# -*- coding: utf-8 -*-
"""
Created on Sat May 14 22:29:37 2022

@author: AbcdRa
"""

import numpy as np
import json

COEF = 11.1

class WNL:
    def psi(x): return (1-x**2)*np.exp(-x**2/2)
    
    def der_psi(x): return (x**3-3*x) * np.exp(-x**2/2)
    
    def f(x): return 1/(1+np.exp(-x))
    
    def der_f(x):
        rf = WNL.f(x)
        return rf*(1-rf)
    
    def to_list(self):
        return [self.n, self.w0.tolist(), self.w1.tolist(),
         self.wu.tolist(), self.ws.tolist(), 
         self.bias.tolist(), self.pv_w0.tolist(),
         self.pv_w1.tolist(), self.pv_wu.tolist(),
         self.pv_ws.tolist(), self.pv_bias.tolist()]
    
    def from_list(l):
        r = WNL(1, l[0],1)
        r.w0 = np.array(l[1])
        r.w1 = np.array(l[2])
        r.wu = np.array(l[3])
        r.ws = np.array(l[4])
        r.bias = np.array(l[5])
        
        r.pv_w0 = np.array(l[6])
        r.pv_w1 = np.array(l[7])
        r.pv_wu = np.array(l[8])
        r.pv_ws = np.array(l[9])
        r.pv_bias = np.array(l[10])
        return r
    
    
    def __init__(self, m, n, l):
        self.n = n
        self.w0 = np.random.rand(n, m)/COEF
        self.w1 = np.random.rand(n, l)/COEF
        self.wu = np.random.rand(n, l, m)
        self.ws = np.random.rand(n, l, m)
        self.bias = np.random.rand(n)
        #self.der_bias = np.ones((n))
        self.pv_w0 = self.w0.copy()
        self.pv_w1 = self.w1.copy()
        self.pv_wu = self.wu.copy()
        self.pv_ws = self.ws.copy()
        self.pv_bias = self.bias.copy()
        
    def predict(self, X):
        return np.array([self.get_y(x) for x in X])
        
    def get_y(self, x):
        s1 = self.w0.dot(x)
        #Используем умножение а не деление
        Z = (x-self.wu)/self.ws
        self.Z = Z
        PsiZ = np.prod(WNL.psi(Z), axis=2)
        self.PsiZ = PsiZ
        s2 = np.sum(self.w1*PsiZ, axis=1)
        self.S = s1 + s2 + self.bias
        self.out = WNL.f(self.S)
        self.der_x = WNL.der_f(self.S)
        return self.out
    
    
    
    def get_eps(self, x, target_y):
        return target_y - self.out
        
    
    def update_w(self, x, eps, lr=0.4, momentum=0.1):
        #predict = self.get_y(x)
        der_f = WNL.der_f(self.S)
        
        der_w0 = np.repeat(x.reshape((1,x.shape[0])), self.n, axis=0)
        dw0 = (-der_w0.T*der_f*eps/self.n).T
        new_w0 = self.w0 - lr*dw0 + momentum*(self.w0 - self.pv_w0)
        
        der_w1 = self.PsiZ
        dw1 = (-der_w1.T*der_f*eps/self.n).T
        new_w1 = self.w1 - lr*dw1 + momentum*(self.w1 - self.pv_w1)

        safe_psi = WNL.psi(self.Z) + (WNL.psi(self.Z) == 0)
        PsiReshape = self.PsiZ.reshape((self.PsiZ.shape[0], self.PsiZ.shape[1], 1))
        almost_der = PsiReshape/safe_psi*WNL.der_psi(self.Z)
        w1_reshape = self.w1.reshape((*self.w1.shape, 1))
        der_wu = -w1_reshape*almost_der/self.ws
        der_wu = np.nan_to_num(der_wu)
        dwu = (-der_wu.T*der_f*eps/self.n).T
        new_wu = self.wu - lr*dwu + momentum*(self.wu - self.pv_wu)
        
        der_ws = der_wu*self.Z
        dws = (-der_ws.T*der_f*eps/self.n).T
        new_ws = self.ws - lr*dws + momentum*(self.ws - self.pv_ws)
        
        dbias = -der_f*eps/self.n
        new_bias = self.bias - lr*dbias + momentum*(self.bias - self.pv_bias)
        
        self.der_x = der_f*(new_w0 - np.sum(der_ws, axis=1)).T
        self.der_x = der_f
        #self.der_x = np.sum(self.der_x, axis=1)
        #self.eps_x = (-eps*der_x.T)
        #self.eps_x = np.sum(self.eps_x, axis=1)
        #self.eps_x = 
        self.pv_bias = self.bias
        self.bias = new_bias
        
        self.pv_w0 = self.w0
        self.w0 = new_w0
        
        self.pv_wu = self.wu
        self.wu = new_wu
        
        self.pv_ws = self.ws
        self.ws = new_ws
        
        self.pv_w1 = self.w1
        self.w1 = new_w1
        return dw0
    

class WNN:
    
    def __init__(self, m, n, l, hn, hm):
        self.hn = hn
        self.hm = hm
        self.layers = [WNL(m, hm, l)]
        self.layers.extend([WNL(hm, hm,l) for i in range(hn)])
        self.layers.append(WNL(hm, n, l))
        
    
    def to_list(self):
        return [self.hn, self.hm, [l.to_list() for l in self.layers]]
    
    def from_list(l):
        r = WNN(1,1,1, l[0], l[1])
        r.layers = [WNL.from_list(i) for i in l[2]]
        return r
    
    def save(self, filename):
        f = open(filename, "w")
        json.dump(self.to_list(), f)
        f.close()
    
    def load(filename):
        f = open(filename, "r")
        wv = WNN.from_list(json.load(f))
        f.close()
        return wv
        
    def get_y(self, x):
        y = x
        for l in self.layers:
            y = l.get_y(y)
        return y
        
    def update_w(self, x, target_y, lr=0.1, momentum=0.0001):
        self.get_y(x)
        eps = self.layers[-1].get_eps(x, target_y)
        prev_outs = self.layers[-2].out
        self.layers[-1].update_w(prev_outs, eps, lr, momentum)
        eps = eps.dot(self.layers[-1].w0)*self.layers[-2].der_x.T
      
        for i in range(self.hn):
            l = self.layers[len(self.layers)-2 - i]
            pl = self.layers[len(self.layers)-3 - i]
            l.update_w(pl.out, eps,lr, momentum)
            eps = eps.dot(l.w0)*pl.der_x.T
            
        self.layers[0].update_w(x, eps, lr, momentum)
            
        
            
    def predict(self, X):
        for l in self.layers:
            X = l.predict(X)
        return X
    
    def fit(self, X, Y, iter_num=1000, learn_rate=0.01, momentum=0.0001):
        i = 0
        while i < iter_num:
            j = np.random.randint(0, len(Y))
            x = X[j]
            y = Y[j]
            self.update_w(x, y, learn_rate, momentum)
            i += 1
    
import matplotlib.pyplot as plt
# W = WNN(2, 3, 6, 1, 20)
# X = np.array([[0,0],[0,1],[1,0],[1,1]])
# Y = np.array([[0,0,0], [1,1,0], [1,1,0], [0,1,1]])
# # X = np.linspace(1,4, 30)
# # Y = np.cos(5*X)+np.random.rand(30)*X/2 + X/2
# # Y = Y.reshape(30,1)
# # X = X.reshape(30,1)
# W.fit(X, Y, 10000, 7.0, 0.01)
# # T = 1/W.predict(X)
# # plt.scatter(X, Y)
# # plt.plot(X, T)
# # plt.show()
# print(W.predict(X))






#Z1 = np.array()
# w0 = np.array([[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# w1 = np.array([[1,0,1,0], [0,1,1,0]])
# wu = np.array([[[0.5, 1, 1.5, 2],[0, 0.5, 0.5, 1]], 
#                [[0.1, 0.01, 0.001, 0.0001], 
#                 [1, 2, 3, 4]]])
# ws = np.array([[[1,0],[0,1],[1,0.5],[0.5,0]],[[1,0],[0,1],[1,0.5],[0.5,0]] ])
# r = x - wu
# r1 = WNL.psi(r)
# r2 = np.prod(r1, axis=2)

