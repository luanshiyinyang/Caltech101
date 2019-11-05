"""
Author: Zhou Chen
Date: 2019/11/4
Desc: 数据加载模块
"""
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示警告
random_seed = 2019


def preprocess(x, y):
    """
    数据预处理
    :param x: 图片路径
    :param y: 图片编码
    :return:
    """
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [224, 224])

    x = tf.image.random_flip_left_right(x)

    # x: [0,255]=>[0,1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=101)

    return x, y


def load_data(desc_path, batch_size):
    """
    加载csv文件读入数据集
    :param desc_path:
    :return:
    """
    df_desc = pd.read_csv(desc_path)
    sample_num = len(df_desc)  # 所有训练数据的样本数目
    images = df_desc['file_id']
    labels = df_desc['label']
    idx = tf.random.shuffle(tf.range(sample_num), seed=random_seed)
    # 按照8:2取训练集和验证集
    train_images, train_labels = tf.gather(images, idx[:int(images.shape[0] * 0.8)]), tf.gather(labels, idx[:int(labels.shape[0]*0.8)])
    valid_images, valid_labels = tf.gather(images, idx[int(images.shape[0] * 0.8):]), tf.gather(labels, idx[int(labels.shape[0]*0.8):])
    db_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    db_valid = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))
    db_train = db_train.shuffle(1000).map(preprocess).batch(batch_size)
    db_valid = db_valid.map(preprocess).batch(batch_size)
    return db_train, db_valid


if __name__ == '__main__':
    load_data("../data/desc.csv", 32)
