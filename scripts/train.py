"""
Author: Zhou Chen
Date: 2019/11/4
Desc: 模型训练
"""
import tensorflow as tf
from model import vgg16, resnet50, densenet121
from data import load_data
from utils import save_pickle
from visualize import plot_history


def train(epochs):

    vgg = vgg16((None, 224, 224, 3), 101)
    resnet = resnet50((None, 224, 224, 3), 101)
    densenet = densenet121((None, 224, 224, 3), 101)
    models = [vgg, resnet, densenet]
    his = []
    for model in models:
        # 创建回调
        es = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0.001,
            patience=5
        )
        lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.2,
            patience=5,
            min_lr=1e-5)
        mc = tf.keras.callbacks.ModelCheckpoint(
            "../models/best_model.h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )
        # 编译模型
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 已经设置了softmax则不需要概率化
                      metrics=['accuracy'])
        db_train, db_test = load_data("../data/desc.csv", 32)
        his.append(model.fit(db_train, validation_data=db_test, validation_freq=1, epochs=epochs, callbacks=[lr, mc]))
    return his


if __name__ == '__main__':
    training_history = train(50)
    plot_history(training_history)
