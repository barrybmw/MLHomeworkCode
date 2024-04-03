#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 19:48:18 2023

@author: liuzhengzuo
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 读取图片数据
def read_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = plt.imread(img_path).ravel() # 拉成一维向量
        images.append(img)
    return np.array(images)

# 读取男生和女生的图片数据
male_images = read_images('/Users/liuzhengzuo/Desktop/PRML/作业/H3/face_data/1')[:300]
female_images = read_images('/Users/liuzhengzuo/Desktop/PRML/作业/H3/face_data/0')[:300]

# 构造标签（1 表示男生，-1 表示女生）
y_male = np.ones(len(male_images))
y_female = -1 * np.ones(len(female_images))

# 构造训练集和测试集
train_male_images = male_images[:250]
train_female_images = female_images[:250]
test_male_images = male_images[250:]
test_female_images = female_images[250:]
X_train = np.concatenate((train_male_images, train_female_images))
y_train = np.concatenate((y_male[:250], y_female[:250]))
X_test = np.concatenate((test_male_images, test_female_images))
y_test = np.concatenate((y_male[250:], y_female[250:]))

# 训练模型
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']
C_values = [0.1, 1, 10, 100]
for kernel_function in kernel_functions:
    for C in C_values:
        svm = SVC(kernel=kernel_function, C=C)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Kernel Function: {kernel_function}, C: {C}, Accuracy: {accuracy}')

        # 找到支持向量
        support_vectors = svm.support_vectors_

        # 在训练集中找到支持向量的下标
        sv_indices = svm.support_

        # 输出支持向量
        print(f'Support Vectors: {sv_indices}')
