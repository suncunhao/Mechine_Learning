#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/17 9:54
# @Author  : sch
# @File    : basic_usage.py

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

import pandas as pd
import numpy as np
import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
