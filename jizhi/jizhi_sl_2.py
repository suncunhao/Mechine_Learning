#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/3 11:43
# @Author  : sch
# @File    : jizhi_sl_2.py
# from:https://jizhi.im/blog/post/sklearntutorial0201

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# 2.1:统计学习：基本设置和预测器对象
# 数据集
# 鸢尾花：Scikit-learn内置的简单数据集范例
# 从scikit-learn库中导入数据集模块datasets
from sklearn import datasets

# 调用datasets的load-iris()方法创建对象，并赋值
iris = datasets.load_iris()

# 将对象iris的data属性赋值给data
data = iris.data

# 输出data的shape属性
print(data.shape)


# 数字数据集：数据变形的距离
digits = datasets.load_digits()
digits.images.shape

# 以灰度图像形式输出-1的像素图
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)

# 为了在scikit中使用，把8*8转换成64的矢量
data = digits.images.reshape((digits.images.shape[0], -1))


# 预测器对象
# 所有预测器都有以数据集（通常为2D数组）为输入参数的fit方法
# estimator.fit(data)

# 预测器参数：所有的预测器参数都可以在实例化的时候指定，或之后修改相应属性
# estimator = Estimator(param1=1, param2=2)


# 2.2:从多维采样中预测输出
# 机器学习解决的问题
# 有监督学习从两个数据集中发现关联：X与y
# Scikit-learn中所有的有监督预测器都提供fit(X, y)方法建立拟合模型
# 使用predict(X)方法，输入无标签采样X，返回预测标签y

# 分类与回归

# 近邻分类与维数灾难
# 分类鸢尾花
# 根据花瓣、萼片的长、宽识别3种不同的鸢尾花
# 导入数据
from sklearn import datasets

# 从数据中导入鸢尾花
iris = datasets.load_iris()

# 将特征数据赋值给iris_X,将标签数据赋值给iris_y
iris_X = iris.data
iris_y = iris.target

# 输出iris_y的不重复元素
np.unique(iris_y)

# k近邻算法
# 近邻算法是最简单的实用分类器：针对每个新采样X_test，在训练集中寻找特征向量最为接近的采样

# 训练集与测试集
# KNN(k nearest neighbors) classification example
np.random.seed(0)

indices = np.random.permutation(len(iris_X))
# permutation:返回一个随机排列
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# 从Scikit-learn库中导入分类器并创建对象knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# 用训练集拟合数据
knn.fit(iris_X_train, iris_y_train)

# 用测试集进行预测
iris_y_predict = knn.predict(iris_X_test)
print('iris_y_predict = ')
print(iris_y_predict)

# 与测试集标签进行对比
iris_y_test
print('iris_y_test = ')
print(iris_y_test)

# 维数灾难
'''
对于一个有效的预测器而言，你需要使相邻采样点之间的距离小于d。在一维情况下，我们需要平均n(1/d)的点
在上述KNN的例子中，如果数据只有一个值为0到1的特征，且有n个采样点，那么新数据总能落在与相邻点不到1/n的距离内
因而当1/n相比异类特征偏差很小的时候，最近邻算法总是有效的

假设特征的数量不是1，而是p，此时我们需要n/(d^p)个点。
比如在一维情况下我们需要10个点，那么p维情况我们需要10^p个点来填充[0,1]空间
随着p的增大，支撑有效预测器的训练点数量将呈指数倍增长
'''

# 线性模型：从回归到稀疏
# 糖尿病数据集——包括442名病人的10个体征变量以及一年后的病情
# 导入数据
diabets = datasets.load_diabetes()

# 划分训练集与测试集，取定20个测试样本
diabets_X_train = diabets.data[:-20]
diabets_X_test = diabets.data[-20:]
diabets_y_train = diabets.target[:-20]
diabets_y_test = diabets.target[-20:]

# 线性回归
# 最简单的线性回归拟合了一个线性模型，通过参数调整一系列参数使得偏差平方和最小
# 线性模型：y = Xβ+ξX  y目标变量 β系数 ξ采样噪音
# Linear regression
from sklearn import linear_model

# 建立线性回归模型对象
regr = linear_model.LinearRegression()
regr.fit(diabets_X_train, diabets_y_train)
print(regr.coef_)

# 偏差平方和
np.mean((regr.predict(diabets_X_test) - diabets_y_test) ** 2)

# 偏差解释：1代表完美，0代表X与y毫无线性关系
score = regr.score(diabets_X_test, diabets_y_test)
print(score)


# 支持向量机SVM
# 线性支持向量机：判别式模型，寻找一组例子建立一个平面，使得两个类别间的空间最大
# 参数C控制正则化，C越小，参与计算的采样点更多
# 支持向量机可以用于regression回归或classification分类
# SVM example
from sklearn import svm

# 创建SVC对象，并使用线性核函数
svc = svm.SVC(kernel ='linear')
svc.fit(iris_X_train, iris_y_train)
# 除了线性核函数，还有(kernel='poly', degree=3)以及(kernel='rbf')

# 练习
# 尝试只用前两个特征，通过SVM将鸢尾花的1、2类别分类，每个类别留出10%为测试集
# 提示：可以在网格上使用decision_function方法获得直观信息
iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 0, :2]
y = y[y != 0]

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 样本容量
n_sample = len(X)

# 产生随机序列用于划分train/test
np.random.seed(0)
order = np.random.permutation(n_sample)
# order就是对X的目录随机排序
X = X[order]
y = y[order].astype(np.float)

end = int(.9 * n_sample)

X_train = X[:end]
y_train = y[:end]
X_test = X[end:]
y_test = y[end:]

# 对不同kernel分别训练模型
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)
    # zorder:控制绘图顺序
    # 输出测试数据图标
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
    #
    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY=np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    # 将计算结果加入图表
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()


