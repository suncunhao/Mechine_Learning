#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/6 9:47
# @Author  : sch
# @File    : Reconginizing_hand_written_digits.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# 载入模块
from sklearn import datasets, svm, metrics
# 读取数据
digits = datasets.load_digits()
# 转换格式，输出前四幅
images_and_labels = list(zip(digits.images, digits.target))
# zip函数：将两个list之间按位置配对
for index, (image, label) in enumerate(images_and_labels[:4]):
# enumerate:遍历函数
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
# axis('off'):不显示坐标
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

n_samples = len(digits.images)
# 样本总数量
data = digits.images.reshape((n_samples, -1))
# digits.images的图像数据是8*8的数组表示的
# reshape进行扁平化处理，这里注意-1的用法

classifier = svm.SVC(gamma=0.001)
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print('Classification report for classifier %s:\n%s\n'
      % (classifier, metrics.classification_report(expected, predicted)))
# 返回数值分别为：准确率（precision）、召回率（recall）和F值（f1-score）

print('Confusion matrix:\n%s' % metrics.confusion_matrix(expected, predicted))
# 计算  混淆矩阵，一定要注意参数位置！先写实际结果后输出预测值
# 关于混淆矩阵：横轴为实际值，纵轴为预测值
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction:%i' % prediction)
plt.show()