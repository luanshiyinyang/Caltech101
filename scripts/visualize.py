"""
Author: Zhou Chen
Date: 2019/11/6
Desc: 可视化训练情况
"""
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')


def plot_history(his):
    vgg_his = his[0]
    resnet_his = his[1]
    densenet_his = his[2]
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(len(vgg_his['accuracy'])), vgg_his['accuracy'])
    plt.plot(np.arange(len(vgg_his['val_accuracy'])), vgg_his['val_accuracy'])
    plt.title("VGG16")

    plt.subplot(1, 3, 2)
    plt.plot(np.arange(len(resnet_his['accuracy'])), resnet_his['accuracy'])
    plt.plot(np.arange(len(resnet_his['val_accuracy'])), resnet_his['val_accuracy'])
    plt.title("ResNet50")

    plt.subplot(1, 3, 3)
    plt.plot(np.arange(len(densenet_his['accuracy'])), densenet_his['accuracy'])
    plt.plot(np.arange(len(densenet_his['val_accuracy'])), densenet_his['val_accuracy'])
    plt.title("DenseNet121")

    plt.savefig("his.png")
    plt.show()

