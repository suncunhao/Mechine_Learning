#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/7 10:22
# @Author  : sch
# @File    : Sparsity_Example_Fitting_only_features_1_and_2.py
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from mpl_toolkits.mplot3d import Axes3D
#
# diabetes = datasets.load_diabetes()
# indices = (0, 1)
#
# X_train = diabetes.data[:-20, indices]
# X_test = diabetes.data[-20:, indices]
# y_train = diabetes.target[:-20]
# y_test = diabetes.target[-20:]
#
# ols = linear_model.LinearRegression()
# ols.fit(X_train, y_train)
#
# # 绘图
# def plot_figs(fig_num, elev, azim, X_train, clf):
#     fig = plt.figure(fig_num, figsize=(4, 3))
#     plt.clf()
#     ax = Axes3D(fig, elev=elev, azim=azim)
#
#     ax.scatter(X_train[:, 0], X_train[:, 1], y_train, x='k', marker='+')
#     ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]),
#                     np.array([[-.1, .15], [-.1, .15]]),
#                     clf.predict(np.array([[-.1, -.1, .15, .15],
#                                           [-.1, .15, -.1, .15]]).T).reshape((2, 2)),
#                     alpha=.5)
#     ax.set_xlabel('X_1')
#     ax.set_ylabel('X_2')
#     ax.set_zlabel('Y')
#     ax.w_xaxis.set_ticklabels([])
#     ax.w_yaxis.set_ticklabels([])
#     ax.w_zaxis.set_ticklabels([])
#
#
# #Generate the three different figures from different views
# elev = 43.5
# azim = -110
# plot_figs(1, elev, azim, X_train, ols)
#
# elev = -.5
# azim = 0
# plot_figs(2, elev, azim, X_train, ols)
#
# elev = -.5
# azim = 90
# plot_figs(3, elev, azim, X_train, ols)
#
# plt.show()
#


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()
indices = (0, 1)

X_train = diabetes.data[:-20, indices]
X_test = diabetes.data[-20:, indices]
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)


# #############################################################################
# Plot the figure
def plot_figs(fig_num, elev, azim, X_train, clf):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)

    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='k', marker='+')
    ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]),
                    np.array([[-.1, .15], [-.1, .15]]),
                    clf.predict(np.array([[-.1, -.1, .15, .15],
                                          [-.1, .15, -.1, .15]]).T
                                ).reshape((2, 2)),
                    alpha=.5)
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('Y')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

#Generate the three different figures from different views
elev = 43.5
azim = -110
plot_figs(1, elev, azim, X_train, ols)

elev = -.5
azim = 0
plot_figs(2, elev, azim, X_train, ols)

elev = -.5
azim = 90
plot_figs(3, elev, azim, X_train, ols)

plt.show()