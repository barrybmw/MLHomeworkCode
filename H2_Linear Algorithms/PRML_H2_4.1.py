#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:02:42 2023

@author: liuzhengzuo
"""

import numpy as np

data = np.loadtxt('breast-cancer-wisconsin.txt')

amount = np.size(data,0) #数据的条目数

#计算m1,m2
m1 = np.zeros(9)
am1 = 0
m2 = np.zeros(9)
am2 = 0
for i in np.arange(0,amount):
    datai = data[i,:]
    if datai[10] == 0:
        m1 = m1 + datai[1:10]
        am1 = am1 + 1
    else:
        m2 = m2 + datai[1:10]
        am2 = am2 + 1
m1 = m1/am1
m2 = m2/am2

#计算Sw
Sw = np.zeros([9,9])
m1 = np.mat(m1)
m2 = np.mat(m2)
for i in np.arange(0,amount):
    datai = data[i,:]
    x = np.mat(datai[1:10])
    if datai[10] == 0:
        Sw = Sw + np.multiply((x-m1).T,x-m1)
    else:
        Sw = Sw + np.multiply((x-m2).T,x-m2)

w = np.linalg.inv(Sw)*(m1-m2).T


#分类
m_1 = w.T*m1.T
m_2 = w.T*m2.T
w0 = -0.5*(m_1+m_2)
print(w0)
T=0

for i in np.arange(0,amount):
    xi = np.mat(data[i,1:10])
    x11 = np.mat(data[i,10])
    gxi = w.T*xi.T+w0
    if gxi >= 0 and x11 == 0:
        T = T+1
    if gxi < 0 and x11 == 1:
        T = T+1

#输出w、准确率
print("w = {}".format(np.round(w,4)))
print("The classification accuracy on the dataset is {:.2f}%.".format(T/amount*100))

