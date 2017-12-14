#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/8 14:59
# @Author  : sch
# @File    : SVM_Margins_Example.py
# SVM边距示例，正则系数C的作用

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

import pandas as pd
import numpy as np

from sklearn import svm

np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# figure number
fignum = 1

# 拟合模型
for name, penalty in (('unreg' ,1), ('reg', 0.05)):
    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)

    # 得到分离超平面
    w = clf.coef_[0]
    a = -w[0] / w[1]                                # 可以理解为斜率
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]        # 超平面,即二维坐标下的直线方程
    # intercept_：求截距

    # 绘制支持向量分离的超空间的平行线
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))    # 二范数
    yy_down = yy - np.sqrt(1 + a ** 2) * margin     # 下平面
    yy_up = yy + np.sqrt(1 + a ** 2) * margin       # 上平面

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    # 标注支持向量

    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')
    # 绘点

    plt.axis('tight')
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    # np.c_很自然的一种用法

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1

plt.show()
