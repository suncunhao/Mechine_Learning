#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/6 13:31
# @Author  : sch
# @File    : Nearest_Neighbors_Classification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

from sklearn import datasets, neighbors
from matplotlib.colors import ListedColormap

n_neighbors = 15

iris = datasets.load_iris()
# 多维数据，切片
X = iris.data[:, :2]
y = iris.target

# 创造色彩图
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # uniform指权重一样，distance指权重为距离的倒数
    # 我们创造一个近邻分类器的距离，并且通过数据进行训练
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 取得x, y两个元素的边界
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # 这样可以保证xx与yy的元素个数一样
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # np.c_按colunms来组合arrays
    # xx.ravel()将多维数组降为一维，注意xx.ravel()有61600个元素
    # yy.ravel()形式为1, 1, 1,...1.02, 1.02, 1.02........
    # 将结果放进一张彩色图中
    # xx.ravel()与yy.ravel()大小相同
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # 类似np.pcolor,对坐标点着色
    # 绘制训练集的点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-class classification (k = %i, weights = '%s')"
            % (n_neighbors, weights))
plt.show()