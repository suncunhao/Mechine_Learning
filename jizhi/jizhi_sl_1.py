#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/1 17:19
# @Author  : sch
# @File    : jizhi_sl_1.py
# from:https://jizhi.im/blog/post/sklearntutorial01

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# 使用sklearn处理数据
from sklearn import datasets

digits = datasets.load_digits()
print(digits.data)
print(digits.target)

# 使用matplotlib绘图
digits = datasets.load_digits()
# 下面我们输出0, 1, 2, 3的8*8点阵图
# 点阵图的数据从datasets读取并存储在digits中
images_and_labels = list(zip(digits.images, digits.target))
# zip函数：将两个list之间按位置配对
for index, (image, label) in enumerate(images_and_labels[:4]):
# enumerate:遍历函数
    plt.subplot(2, 4, index + 1 )
# 分子图绘制
    plt.axis('off')
# 不显示坐标尺寸
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
# camp：色彩
# interpolation：通常图片都是由RGB组成，一块一块的，这里想把某块显示成一种颜色，则需要调用interpolation='nearest'参数
    plt.title('Training:%i' % label)

# 选择分类器并进行设置、训练和预测
# SVM分类器构建
from sklearn import svm
digits = datasets.load_digits()
# 建立SVM分类器
clf = svm.SVC(gamma=0.001, C=100.)
# 使用训练数据对分类器进行训练，将会返回分类器的某些参数设置
clf.fit(digits.data[:-1], digits.target[:-1])
# skleran的训练过程使用clf.fit(train, target)方法

# 用于计算的部分代码已被隐藏，以下是用于预测的未知数据
# 你可以改变这个数据中的数字，但必须保证数组元素个数为64，否则将会出错
test = [0, 0, 10, 14, 8, 1, 0, 0,
        0, 2, 16, 14, 6, 1, 0, 0,
        0, 0, 15, 15, 8, 15, 0, 0,
        0, 0, 5, 16, 16, 10, 0, 0,
        0, 0, 12, 15, 15, 12, 0, 0,
        0, 4, 16, 6, 4, 16, 6, 0,
        0, 8, 16, 10, 8, 16, 8, 0,
        0, 1, 8, 12, 14, 12, 1, 0]
print("对图片的预测结果为：")
print(clf.predict(np.array(test).reshape(1, 64)))
# 这里会出错，主要原因是工具包版本问题
# 根据提示修改（加上.reshape(1, -1)）
plt.imshow(np.array(test).reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
plt.axis('off')
# 显示test图片

# 查看和评估分类效果
from sklearn import metrics
digits = datasets.load_digits()
# 以下是分类器构建过程，也可以尝试修改分类器参数数值
clf = svm.SVC(gamma=0.001, C=100.)
# 选取数据集前500条数据作为训练集
clf.fit(digits.data[:500], digits.target[:500])

# 选取数据集后1000条数据作为测试集
expected = digits.target[-1000:]
predicted = clf.predict(digits.data[-1000:])
print('分类器预测结果评估：\n%s\n'
      % (metrics.classification_report(expected, predicted)))
# 返回数值分别为：准确率（precision）、召回率（recall）和F值（f1-score）

# 练习
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(train, target)
predicted = clf.predict(test)