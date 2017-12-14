#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/29 17:12
# @Author  : sch
# @File    : Exploring_and_Transforming_Data.py
# @Book    :Building Machine Learning Projects with TensorFlow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# sess = tf.Session()
tens1 = tf.constant([[[1, 2], [2, 3]], [[3, 4], [5, 6]]])
print(sess.run(tens1)[1, 1, 0])

