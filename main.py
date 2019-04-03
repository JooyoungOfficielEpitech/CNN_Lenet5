import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from os import system
import numpy as np

dataset_class = sys.argv[1]
epoch = sys.argv[2]

if (dataset_class == "cifar10"):
    from tensorflow.keras.datasets.cifar10 import load_data
    from include.model_lenet import model
else:
    from tensorflow.keras.datasets.mnist import load_data
    from include.model_lenet_mnist import model

(x_train, y_train), (x_test, y_test) = load_data()

if (dataset_class == "mnist"):
    x_train = x_train.reshape([-1, 28, 28, 1])
    x_test = x_test.reshape([-1, 28, 28, 1])

print(x_train.shape)print(y_train.shape)

def one_hot_encoder(y):
    ret = np.zeros(len(y) * 10)
    ret = ret.reshape([-1, 10])
    for i in range(len(y)):
        ret[i][y[i]] = 1
    return (ret)

def random_batch(x, y, size):
    idx = np.random.randint(len(x), size=size)
    batch_x = []
    batch_y = []
    for i in range(size):
        batch_x.append(x[idx[i]])
        batch_y.append(y[idx[i]])
    return (batch_x, batch_y)

y_train_cls = one_hot_encoder(y_train)
y_test_cls = one_hot_encoder(y_test)

train_cost = []
test_acc = []

x, y, dr, loss, optimizer, accuracy = model()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(int(epoch)):
        batch_x, batch_y = random_batch(x_train, y_train_cls, 20)
        cost, _ = sess.run([accuracy, optimizer], feed_dict={x:batch_x, y:batch_y, dr:0.8})
        system("clear")
        print("The acc in epoch " , i, " is ", int(cost * 100), "%")
        print("learnig pourcentage : ", int(i * 100 / int(epoch)), "%")
        if (i % (int(epoch) / 100) == 0):
            train_cost.append(cost)
            batch_x, batch_y = random_batch(x_test, y_test_cls, 20)
            acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y, dr:1})
            test_acc.append(acc)
    plt.plot(test_acc, label='train')
    plt.plot(train_cost, label='test')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
   # plt.savefig('./test/droprate' + '_train_test_acc_' + dataset_class + '_' + epoch + ".png")
