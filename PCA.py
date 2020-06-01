# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:04:39 2020

@author: 刘志伟
"""

import matplotlib.pyplot as plt                 #加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA           #加载PCA算法包
from sklearn.datasets import load_iris          #加载数据集

data=load_iris()                                #载入数据
y=data.target                                   #数据的label
x=data.data                                     #数据的属性
pca=PCA(n_components=2)                         #加载PCA算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(x)                  #对样本进行降维

red_x,red_y=[],[]                               #建立空列表，为后期循环做准备
blue_x,blue_y=[],[]
green_x,green_y=[],[]


for i in range(len(reduced_x)):                 #将压缩后的数据按照label分开，为后期画图做准备
    if y[i] ==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])

    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])

    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

#可视化
plt.scatter(red_x,red_y,c='r',marker='x')       #将三类降维后的数据画在一副图中
plt.scatter(blue_x,blue_y,c='b',marker='.')
plt.scatter(green_x,green_y,c='g',marker='D')
plt.show()