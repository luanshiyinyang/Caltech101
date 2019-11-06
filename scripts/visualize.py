"""
Author: Zhou Chen
Date: 2019/11/6
Desc: 可视化训练情况
"""
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_history(his):
    history = his.history()