"""
Author: Zhou Chen
Date: 2019/11/4
Desc: 模型训练
"""
import tensorflow as tf
from model import vgg16, resnet50, densenet121
from data import load_data


def train(epochs):

    vgg = vgg16((None, 224, 224, 3), 101)
    resnet = resnet50((None, 224, 224, 3), 101)
    densenet = densenet121((None, 224, 224, 3), 101)
    models = [vgg, resnet, densenet]
    his = []
    for model in models:
        # 创建回调
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.001,
            patience=5
        )
        lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-5)
        # 编译模型
        model.compile(optimizer=tf.optimizers.Adam(lr=1e-3),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        db_train, db_test = load_data("../data/desc.csv", 32)
        his.append(model.fit(db_train, validation_data=db_test, validation_freq=1, epochs=epochs, callbacks=[es, lr]))
    return his


if __name__ == '__main__':
    train(1)