import tensorflow as tf
import numpy as np

def model():
    img_size = 32
    color = 3
    num_classes = 10
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, color])
    y = tf.placeholder(tf.float64, shape=[None, num_classes])
    drop_rate = tf.placeholder(tf.float32)
    
    w_conv_1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 6], stddev=5e-2))
    b_conv_1 = tf.Variable(tf.constant(0.1, shape=[32, 32, 6]))
    h_conv_1 = tf.nn.relu(tf.nn.conv2d(x, filter=w_conv_1, strides=[1, 1, 1, 1], padding='SAME'))

    pool1 = tf.nn.max_pool(h_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    w_conv_2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 12], stddev=5e-2))
    b_conv_2 = tf.Variable(tf.constant(0.1, shape=[12]))
    h_conv_2 = tf.nn.relu(tf.nn.conv2d(h_conv_1, filter=w_conv_2, strides=[1, 1, 1, 1], padding='SAME'))

    pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_conv_3 = tf.Variable(tf.truncated_normal(shape=[5, 5, 12, 12], stddev=5e-2))
    h_conv_3 = tf.nn.relu(tf.nn.conv2d(h_conv_2, w_conv_3, strides=[1, 1, 1, 1], padding='SAME'))

    conv_layers_1 = tf.layers.conv2d(x, 6, [5, 5], padding="SAME", activation=tf.nn.relu)

    pool_1 = tf.nn.max_pool(conv_layers_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv_layers_2 = tf.layers.conv2d(pool_1, 12, [5, 5], padding='SAME', activation=tf.nn.relu)

    pool_2 = tf.nn.max_pool(conv_layers_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    flat = tf.reshape(h_conv_1, [-1, 32 * 32 * 6])

    w_fc_2 = tf.Variable(tf.truncated_normal(shape=[32 * 32 * 6, 200]))
    w_fc_2 = tf.scalar_mul(np.sqrt(200 / 2) , w_fc_2)
    b_fc_2 = tf.Variable(tf.constant(0.1, shape=[200]))
    h_fc_2 = tf.nn.relu(tf.matmul(flat, w_fc_2) + b_fc_2)

    drop_4 = tf.nn.dropout(h_fc_2, keep_prob=drop_rate)
    
    w_out = tf.Variable(tf.truncated_normal(shape=[200, 10]))
    w_out = tf.scalar_mul(np.sqrt(10 / 2), w_out)
    b_out = tf.Variable(tf.constant(0.1, shape=[10]))
    h_out = tf.nn.sigmoid(tf.matmul(drop_4, w_out) + b_out)

    y_pred_cls = tf.argmax(h_out, axis=1)

    correct_ans = tf.equal(y_pred_cls, tf.reshape(tf.argmax(y, axis=1), [-1]))

    accuracy = tf.reduce_mean(tf.cast(correct_ans, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=tf.cast(h_out, tf.float32)))
    optimizer = tf.train.RMSPropOptimizer(1e-5).minimize(loss)

    return (x, y, drop_rate, loss, optimizer, accuracy)
