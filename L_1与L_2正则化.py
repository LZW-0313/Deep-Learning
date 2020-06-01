# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:59:06 2020

@author: lx
"""
##导入相关算法库
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

##载入boston房价数据集
boston = load_boston() 

##标准化处理
scaler = StandardScaler()
X = scaler.fit_transform(boston["data"])

Y = boston["target"]

names = boston["feature_names"]

##传统线性回归模型
lr = LinearRegression()
lr.fit(X,Y)             #fit()默认loss为RSS,算法为GD

##loss中加入l1正则项的lasso
lasso = Lasso(alpha=.3)
lasso.fit(X, Y)

##loss中加入l2正则项的岭回归
ridge = Ridge(alpha=.3)
ridge.fit(X,Y)

##分别输出三种方法所得到的参数值
print("Linear model:", lr.coef_)
print('Lasso model:',lasso.coef_)
print("Ridge model:",ridge.coef_)












