"""
Author: Zhou Chen
Date: 2019/11/4
Desc: 模型训练
"""
import tensorflow as tf
from model import vgg16, resnet50, densenet121
from data import load_data
from visualize import plot_history


def load_db(batch_size):
    """
    加载数据集
    :param batch_size:
    :return:
    """
    db_train, db_test = load_data("../data/desc.csv", batch_size)
    return db_train, db_test


def train(epochs):
    """
    训练模型
    :param epochs:
    :return:
    """

    vgg = vgg16((None, 224, 224, 3), 102)
    resnet = resnet50((None, 224, 224, 3), 102)
    densenet = densenet121((None, 224, 224, 3), 102)
    models = [vgg, resnet, densenet]
    train_db, valid_db = load_db(32)
    his = []
    for model in models:
        variables = model.trainable_variables
        optimizers = tf.keras.optimizers.Adam(1e-4)
        for epoch in range(epochs):
            # training
            total_num = 0
            total_correct = 0
            training_loss = 0
            for step, (x, y) in enumerate(train_db):
                print(y.shape)
                with tf.GradientTape() as tape:
                    # train
                    out = model(x)
                    loss = tf.losses.categorical_crossentropy(y, out, from_logits=False)
                    loss = tf.reduce_mean(loss)
                    training_loss += loss
                    grads = tape.gradient(loss, variables)
                    optimizers.apply_gradients(zip(grads, variables))
                    # training accuracy
                    y_pred = tf.cast(tf.argmax(out, axis=1), dtype=tf.int32)
                    y_true = tf.cast(tf.argmax(y, axia=1), dtype=tf.int32)
                    correct = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), dtype=tf.int32))
                    total_num += x.shape[0]
                    total_correct += int(correct)
            training_accuracy = total_correct / total_num

            # validation
            total_num = 0
            total_correct = 0
            for (x, y) in valid_db:
                out = model(x)
                y_pred = tf.argmax(out, axis=1)
                y_pred = tf.cast(y_pred, dtype=tf.int32)
                y_true = tf.argmax(y, axia=1)
                y_true = tf.cast(y_true, dtype=tf.int32)
                correct = tf.cast(tf.equal(y_pred, y_true), dtype=tf.int32)
                correct = tf.reduce_sum(correct)
                total_num += x.shape[0]
                total_correct += int(correct)
            validation_accuracy = total_correct / total_num
            print("epoch:{}, training loss:{:.4f}, training accuracy:{:.4f}, validation accuracy:{:.4f}".format(epoch, training_loss, training_accuracy, validation_accuracy))
            his.append({'accuracy': training_accuracy, 'val_accuracy': validation_accuracy})
    return his


if __name__ == '__main__':
    training_history = train(50)
    plot_history(training_history)
