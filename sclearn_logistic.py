#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/2 9:38
# @Author  : sch
# @File    : sclearn_logistic.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

from sklearn import datasets
from sklearn.model_selection import train_test_split
# 原文使用的CV模块，已被合并入model_selection
#
# iris = datasets.load_iris()
# X = iris.data[:, [2, 3]]
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)
#
# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))
#
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(C=1000., random_state=0)
# lr.fit(X_train_std, y_train)
# lr.predict_proba(X_combined_std, y_combined)
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.show()


def sigmoid(h):
    return 1.0 / (1.0 + np.exp(-h))

h = np.arange(-10, 10, 0.1) # 定义x的范围，像素为0.1
s_h = sigmoid(h) # sigmoid为上面定义的函数
plt.plot(h, s_h)
plt.axvline(0.0, color='k') # 在坐标轴上加一条竖直的线，0.0为竖直线在坐标轴上的位置
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted') # 加水平间距通过坐标轴
plt.axhline(y=0.5, ls='dotted', color='k') # 加水线通过坐标轴
plt.yticks([0.0, 0.5, 1.0]) # 加y轴刻度
plt.ylim(-0.1, 1.1) # 加y轴范围
plt.xlabel('h')
plt.ylabel('$S(h)$')
plt.show()

print(__doc__)

pd.read_clipboard()
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
