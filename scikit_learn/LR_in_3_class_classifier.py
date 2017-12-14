#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/3 16:57
# @Author  : sch
# @File    : LR_in_3_class_classifier.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

from sklearn import linear_model, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].

'''
在直角坐标系中，x表示花萼长度，y表示花萼宽度。每个点的坐标就是(x,y)。 
先取X二维数组的第一列（长度）的最小值，最大值和步长h生成数组, 再取X二维数组的第二列（宽度）的最小值，最大值和步长h生成数组.
然后用meshgrid函数生成两个网格矩阵xx和yy
'''
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# xx.ravel() 和 yy.ravel() 是将两个矩阵（二维数组）都变成一维数组的意思（其实是视图，并没有复制数据）,由于两个矩阵大小相等，因此两个一维数组大小也相等。
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()

# 讲解：http://blog.csdn.net/csfreebird/article/details/52744037