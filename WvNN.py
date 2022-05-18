# -*- coding: utf-8 -*-
"""
Created on Thu May 12 01:13:57 2022

@author: AbcdRa
"""

import numpy as np
import matplotlib.pyplot as plt
import json

#https://core.ac.uk/download/pdf/10637269.pdf

np.seterr(divide='ignore', invalid='ignore')
COEF =1
class Wavelon:
    #m - кол-во входов
    #l - кол-во HU
    
    def mother_wv(x): return (1-x**2)*np.exp(-x**2/2)
    
    def mother_wv_der(x): return (x**3-3*x) * np.exp(-x**2/2)
    
    #def multidim_wv(x, j, wu, ws):
    #    return np.prod(Wavelon.mother_wv((x-wu.T[j])/ws.T[j]))
    
    def multidim_wv(x, wu, ws):
        return np.prod(Wavelon.mother_wv((x-wu.T)/ws.T), axis=1)
      
    
    def __init__(self, m, l):
        global COEF
        #Вес смещения
        self.bias = np.random.random()
        self.pv_bias = self.bias
        self.der_bias = 1
        #Линейные веса входа
        self.w0 = np.random.random(m)/COEF
        self.pv_w0 = self.w0.copy()
        #Веса сдвигов
        self.wu = np.random.rand(m,l)*2
        self.pv_wu = self.wu.copy()
        #Веса масштаба
        self.ws = np.random.rand(m,l)
        self.pv_ws = self.ws.copy()
        #Веса вейвлета
        self.w1 = np.random.random(l)/COEF
        self.pv_w1 = self.w1.copy()
        
    
    def get_new_w(ep, curr_w, prev_w, derr_w, learn_rate, momentum):
        return curr_w+learn_rate*ep*derr_w+(curr_w-prev_w)*momentum
        
    def update_der(self, x):
        Zij = (x-self.wu.T)/self.ws.T
        self.der_w0 = x
        self.der_w1 = np.prod(Wavelon.mother_wv(Zij), axis=1)
        psi_z = Wavelon.mother_wv(Zij)
        safe_psi_z = psi_z + (psi_z==0)
        der_wu = -Wavelon.mother_wv_der(Zij)/safe_psi_z
        #der_wu = np.nan_to_num(der_wu)
        der_wu = der_wu.T * self.der_w1*self.w1/self.ws
        self.der_wu = der_wu
        self.der_ws = self.der_wu*Zij.T
        
    def update_der_x(self):
        self.derr_x = (self.w0-np.sum(self.der_wu, axis=1))
        
    
    def set_ep(self, ep):
        self.ep = ep
        
    def to_list(self):
        return [ self.bias, self.pv_bias,
        self.w0.tolist(), self.pv_w0.tolist(), self.wu.tolist(),
        self.pv_wu.tolist(), self.ws.tolist(), self.pv_ws.tolist(),
        self.w1.tolist(), self.pv_w1.tolist()]
    
    def from_list(wvList):
        s = Wavelon(0,0)
        s.bias, s.pv_bias, s.w0, s.pv_w0, s.wu, s.pv_wu, s.ws, s.pv_ws, s.w1, s.pv_w1 = wvList
        s.w0 = np.array(s.w0)
        s.pv_w0 = np.array(s.pv_w0)
        s.w1 = np.array(s.w1)
        s.pv_w1 = np.array(s.pv_w1)
        s.wu = np.array(s.wu)
        s.pv_wu = np.array(s.pv_wu)
        s.ws = np.array(s.ws)
        s.pv_ws = np.array(s.pv_ws)
        return s
        
    def update_w(self, x, learn_rate=0.005, momentum=0.05):
        self.update_der(x)
        ep = self.ep
        new_w0 = Wavelon.get_new_w(ep,self.w0,self.pv_w0,self.der_w0,learn_rate, momentum)
        new_w1 = Wavelon.get_new_w(ep,self.w1,self.pv_w1,self.der_w1,learn_rate, momentum)
        new_wu = Wavelon.get_new_w(ep,self.wu,self.pv_wu,self.der_wu,learn_rate, momentum)
        new_ws = Wavelon.get_new_w(ep,self.ws,self.pv_ws,self.der_ws,learn_rate, momentum)
        new_b = Wavelon.get_new_w(ep,self.bias,self.pv_bias,self.der_bias,learn_rate, momentum)
        self.pv_bias = self.bias
        self.bias = new_b
        self.pv_w0 = self.w0
        self.w0 = new_w0
        self.pv_w1= self.w1 
        self.w1 = new_w1
        self.pv_wu = self.wu
        self.wu =  new_wu
        self.pv_ws = self.ws
        self.ws = new_ws
    
    def update_ep(self, x, target_y):
        self.ep = target_y - self.get_y(x)
    
    
        
    def get_y(self, x):
        s1 = np.sum(self.w1 * Wavelon.multidim_wv(x, self.wu, self.ws))
        s2 = np.sum(self.w0*x)
        self.out = s1 + s2 + self.bias
        return self.out
    
    def getY(self, X):
        v = np.vectorize(self.get_y)
        return v(X)


class WvLayer:
    #m - кол-во входов
    #n - кол-во выходов
    #l - кол-во HU
    def __init__(self, m, n, l):
        self.Wvs = [Wavelon(m, l) for i in range(n)]
        self.n = n
    
    def update_w(self, x, learn_rate=0.01, momentum=0.001): 
        eps = self.eps
        for i in range(len(eps)):
            self.Wvs[i].set_ep(eps[i])
            self.Wvs[i].update_w(x,learn_rate, momentum)
    
    
    def get_input_err(self,x):
        err = []
        for i in range(self.n):
            wv = self.Wvs[i]
            wv.update_der(x)
            wv.update_der_x()
            err.append(-self.eps[i]*wv.derr_x)
        return np.sum(err,axis=0)
    
    def set_eps(self, eps):
        self.eps = eps
        for i in range(len(eps)):
            self.Wvs[i].set_ep(eps[i])
    
    def get_Y(self, x):
        self.y = np.array([self.Wvs[i].get_y(x) for i in range(self.n)])
        return self.y
    
    def predict(self, X):
        return np.array([self.get_Y(x) for x in X])
        
    def get_outs(self):
        self.outs = np.array([wv.out for wv in self.Wvs])
        return self.outs
    
    def to_list(self):
        return [wv.to_list() for wv in self.Wvs]
    
    def from_list(wlList):
        l = WvLayer(1, len(wlList), 1)
        for i in range(len(wlList)):
            l.Wvs[i] = Wavelon.from_list(wlList[i])
        return l
    
    def save(self, filename):
        f = open(filename, "w")
        json.dump(self.to_list(), f)
        f.close()
    
    def load(filename):
        f = open(filename, "r")
        wv = WvLayer.from_list(json.load(f))
        f.close()
        return wv
        
    def fit(self, X, Y, iter_num=1000, learning_rate=0.001, momentum=0.0001, is_log=False):
        for i in range(iter_num):
            #for j in range(len(Y)):
                j = np.random.randint(0, len(Y))
                x = X[j]
                y = Y[j]
                self.update_eps(x, y)
                if is_log and i%100==0: print(np.sum(self.eps**2))
                self.update_w(x)        
    
    def update_eps(self, x, target_y):
        self.eps = target_y - self.get_Y(x)
        return self.eps
    

class WvNN:
    
    #
    def __init__(self, m, n, hlshape, HU):
        self.numL = hlshape[0] + 2
        self.Ls = [WvLayer(m, hlshape[1], HU)]
        for i in range(hlshape[0]):
            self.Ls.append(WvLayer(hlshape[1],hlshape[1],HU))
        self.Ls.append(WvLayer(hlshape[1], n, HU))
        
    def predict(self, x):
        y = x
        for layer in self.Ls:
            y = layer.get_Y(y)
        return y
    
    def to_list(self):
        l = [self.Ls[i].to_list() for i in range(self.numL)]
        return l
    
    def from_list(l):
        o = WvNN(1,1,(1,1),1)      
        o.Ls = [WvLayer.from_list(x) for x in l]
        o.numL = len(o.Ls)
        return o
    
    def predictY(self, X):
        return np.array([self.predict(x) for x in X])
    
    def update_w(self, x, eps, lr=0.001, m=0.0001):
        for i in range(self.numL-1):
            l = self.Ls[self.numL - 1- i]
            pvl = self.Ls[self.numL - 2- i]
            l.set_eps(eps)
            eps = l.get_input_err(pvl.get_outs()) 
            l.update_w(pvl.get_outs(), lr, m)
            
        self.Ls[0].set_eps(eps)
        self.Ls[0].update_w(x, lr, m)
            
    def get_eps(self, x, target_y):
        return target_y - self.predict(x)
    
    def fit(self, X, Y, iter_num=1000, learning_rate=0.001, momentum=0.0001):
        for i in range(iter_num):
            #for j in range(len(Y)):
                j = np.random.randint(0, len(Y))
                x = X[j]
                y = Y[j]
                eps = self.get_eps(x, y)
                self.update_w(x, eps)
    
    def save(self, filename):
        f = open(filename, "w")
        json.dump(self.to_list(), f)
        f.close()
    
    def load(filename):
        f = open(filename, "r")
        wv = WvNN.from_list(json.load(f))
        f.close()
        return wv
        
            
        
            

X = np.linspace(1,4, 60)
Y = np.cos(5*X)+np.random.rand(60)*X/2 + X/2
#Wv = WvNN(1, 1, (1, 2), 15)
Wv = Wavelon(1, 20)

for i in range(3000):
    j=np.random.randint(0, 60)
    x = X[j]
    y = Y[j]
    Wv.update_ep(x, y)
    Wv.update_w(x, 0.009, 0.00001)
    if (i+1) % 500 == 0:
        T = Wv.getY(X)
        plt.scatter(X, Y)
        plt.plot(X, T, color="g")
        plt.show()





        

    
    