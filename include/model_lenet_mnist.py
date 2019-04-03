import tensorflow as tf
import numpy as np

def model():
    img_size = 28
    color = 1
    num_classes = 10
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, color])
    y = tf.placeholder(tf.float64, shape=[None, num_classes])
    drop_rate = tf.placeholder(tf.float32)
    
    w_conv_1 = tf.Variable(tf.truncated_normal(shape=[5, 5, color, 6], stddev=5e-2))
    b_conv_1 = tf.Variable(tf.constant(0.1, shape=[6]))
    h_conv_1 = tf.nn.relu(tf.nn.conv2d(x, filter=w_conv_1, strides=[1, 2, 2, 1], padding='VALID') + b_conv_1)

    w_conv_2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 12], stddev=5e-2))
    b_conv_2 = tf.Variable(tf.constant(0.1, shape=[12]))
    h_conv_2 = tf.nn.relu(tf.nn.conv2d(h_conv_1, filter=w_conv_2, strides=[1, 2, 2, 1], padding='VALID') + b_conv_2)

    flat = tf.reshape(h_conv_2, [-1, 4 * 4 * 12])

    w_fc_1 = tf.Variable(tf.truncated_normal(shape=[4 * 4 * 12, 200], stddev=5e-2))
    b_fc_1 = tf.Variable(tf.constant(0.1, shape=[200]))
    h_fc_1 = tf.nn.relu(tf.matmul(flat, w_fc_1) + b_fc_1)

    drop_1 = tf.nn.dropout(h_fc_1, keep_prob=drop_rate)
    
    w_fc_2 = tf.Variable(tf.truncated_normal(shape=[200, num_classes], stddev=5e-2))
    b_fc_2 = tf.Variable(tf.constant(0.1, shape=[10]))
    h_fc_2 = tf.matmul(drop_1, w_fc_2) + b_fc_2

    h_out = tf.nn.sigmoid(h_fc_2)
    
    y_pred_cls = tf.argmax(h_out, axis=1)

    correct_ans = tf.equal(y_pred_cls, tf.reshape(tf.argmax(y, axis=1), [-1]))

    accuracy = tf.reduce_mean(tf.cast(correct_ans, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=tf.cast(h_out, tf.float32)))

    optimizer = tf.train.RMSPropOptimizer(1e-4).minimize(loss)
    
    return (x, y, drop_rate, loss, optimizer, accuracy)
