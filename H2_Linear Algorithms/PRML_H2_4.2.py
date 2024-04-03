#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 20:01:22 2023

@author: liuzhengzuo
"""

import numpy as np
import matplotlib.pyplot as plt


#0、数据导入和处理
data = np.loadtxt('breast-cancer-wisconsin.txt')
amount = np.size(data,0)-100 #数据的条目数

#1、随机选取初始参数
w00 = np.array([-1.30335854e-03,-8.98241234e-04, #参考以上一问中w的结果
              -6.41128962e-04,-3.38746967e-04,
              -4.13950521e-04,-1.86464024e-03,
              -7.87745282e-04,-7.61334567e-04,-4.01693948e-05])
wa = np.average(w00) #计算w各个分量的均值
wv = np.std(w00) #计算w各个分量的标准差
w0 = np.random.normal(wa,wv,9) #根据上面计算的均值和标准差随机生成各个分量独立同高斯（正态）分布的w0
b0 = 0

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



#2、计算损失函数
def sig(w,b,x): #定义sigmoid函数
    inn = np.dot(w,x)+b
    sig = 1/(1+np.exp(-inn))
    return sig

L0 = 0 #L为损失函数值
for i in np.arange(0,amount):
    xi = data[i,1:10]
    yi = data[i,10]
    Li = 0.5*(sig(w0,b0,xi)-yi)**2
    L0 = L0 + Li



#3、计算梯度
dl_dw = np.zeros(9)
dl_db = 0
for i in np.arange(0,amount):
    xi = data[i,1:10]
    yi = data[i,10]
    dl_dwi = (sig(w0,b0,xi)-yi)*sig(w0,b0,xi)*(1-sig(w0,b0,xi))*xi
    dl_dbi = (sig(w0,b0,xi)-yi)*sig(w0,b0,xi)*(1-sig(w0,b0,xi))
    dl_dw = dl_dw + dl_dwi
    dl_db = dl_db + dl_dbi



#4、梯度下降
rho = 0.0001 #设置步长，通过损失函数下降情况调整步长
w1 = w0 - rho*dl_dw
b1 = b0 - rho*dl_db



#5、再次计算损失函数
L1 = 0 #L为损失函数值
for i in np.arange(0,amount):
    xi = data[i,1:10]
    yi = data[i,10]
    Li = 0.5*(sig(w1,b1,xi)-yi)**2
    L1 = L1 + Li



#6、进行重复迭代
w = w1
b = b1
rho = 0.00001
for i in np.arange(0,10):
    #计算当前梯度
    dl_dw = np.zeros(9)
    dl_db = 0
    for i in np.arange(0,amount):
        xi = data[i,1:10]
        yi = data[i,10]
        dl_dwi = (sig(w0,b0,xi)-yi)*sig(w0,b0,xi)*(1-sig(w0,b0,xi))*xi
        dl_dbi = (sig(w0,b0,xi)-yi)*sig(w0,b0,xi)*(1-sig(w0,b0,xi))
        dl_dw = dl_dw + dl_dwi
        dl_db = dl_db + dl_dbi
    #计算新的w和b
    w = w - rho*dl_dw
    b = b - rho*dl_db
    #计算损失函数
    L = 0 #L为损失函数值
    for i in np.arange(0,amount):
        xi = data[i,1:10]
        yi = data[i,10]
        Li = 0.5*(sig(w,b,xi)-yi)**2
        L = L + Li
    


#分类
T=0

for i in np.arange(amount,amount+100):
    xi = data[i,1:10]
    yi = data[i,10]
    if sig(w,b,xi) < 0.64 and yi == 0:
        T = T+1
    if sig(w,b,xi) > 0.64 and yi == 1:
        T = T+1
        
Iter = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
Loss = np.array([73.38631294211868,67.30205161722063,66.94952019656954,66.64385394912787,66.38443109244459,66.17045464894073,66.00097103502243,65.87488907961084,65.79099908905998,65.74799161736927,65.74447564774903,65.77899594139363])

print("The classification accuracy on the dataset is {:.2f}%.".format(T/100*100))
plt.plot(Iter,Loss)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.rcParams['savefig.dpi'] = 1000 #图片像素
plt.rcParams['figure.dpi'] = 1000 #分辨率

cos = np.dot(w,w00)/(np.sqrt(np.dot(w,w))*np.sqrt(np.dot(w00,w00)))
print("The cosine between two w in section 1 and 2 is {:.2f}.".format(cos))










