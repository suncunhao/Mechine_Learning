#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/1 10:21
# @Author  : sch
# @File    : CrossValidation.py
# @Link    : http://scikit-learn.org/dev/modules/cross_validation.html#cross-validation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# Cross Validation
from sklearn import cross_validation, datasets, svm
from sklearn.ensemble import RandomForestClassifier
iris = datasets.load_iris()
# iris.data.shape, iris.target.shape
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=0
)
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# K-Flod
from sklearn.model_selection import KFold
X = ['a', 'b', 'c', 'd']
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print('%s %s' %(train, test))

# Repeated K-Fold
