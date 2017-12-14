#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/8 16:10
# @Author  : sch
# @File    : Plot_different_SVM_classifiers_in_the_iris_dataset.py

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

import pandas as pd
import numpy as np

from sklearn import datasets, svm

def make_meshgrid(x, y, h=0.02):
    x_min, x_max = x.min() - 1, x.max() + 1
