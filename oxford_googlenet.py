import tensorflow as tf
import numpy as np
import oxflower17

batch_size = 8
test_size = 136

# Create some wrappers
def conv2d(x, W, b, strides):  # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k, strides):  # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding='SAME')


def avgpool2d(x, k, strides):  # AveragePool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding='SAME')


def local_response_normalization(incoming, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75):
    return tf.nn.lrn(incoming, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)

###### Create NN ###########
def model (x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1,227,227,3])

    conv1_7_7 = conv2d(X, weights['w_c1_77'], biases['b_c1_77'], strides=2)

    pool1_3_3 = maxpool2d(conv1_7_7, k=3, strides=2)
    pool1_3_3 = local_response_normalization(pool1_3_3)

    conv2_1_1 = conv2d(pool1_3_3, weights['w_c2_11'], biases['b_c2_11'], strides=1)
    conv2_3_3 = conv2d(conv2_1_1, weights['w_c2_33'], biases['b_c2_33'], strides=1)
    conv2_3_3 = local_response_normalization(conv2_3_3)

    pool2_3_3 = maxpool2d(conv2_3_3, k=3, strides=2)

    # Inception module (3a)
    inception_3a_1_1 = conv2d(pool2_3_3, weights['w_inception_3a_11'], biases['b_inception_3a_11'], strides=1)
    inception_3a_3_3_reduce = conv2d(pool2_3_3, weights['w_inception_3a_33_reduce'], biases['b_inception_3a_33_reduce'], strides=1)
    inception_3a_3_3 = conv2d(inception_3a_3_3_reduce, weights['w_inception_3a_33'], biases['b_inception_3a_33'], strides=1)
    inception_3a_5_5_reduce = conv2d(pool2_3_3, weights['w_inception_3a_55_reduce'], biases['b_inception_3a_55_reduce'], strides=1)
    inception_3a_5_5 = conv2d(inception_3a_5_5_reduce, weights['w_inception_3a_55'], biases['b_inception_3a_55'], strides=1)
    inception_3a_maxpool = maxpool2d(pool2_3_3, k=3, strides=1)
    inception_3a_maxpool_reduce = conv2d(inception_3a_maxpool, weights['w_inception_3a_mp_reduce'], biases['b_inception_3a_mp_reduce'], strides=1)

    inception_3a_concat = tf.concat(3, [inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_maxpool_reduce])
    # Inception module (3b)
    inception_3b_1_1 = conv2d(inception_3a_concat, weights['w_inception_3b_11'], biases['b_inception_3b_11'], strides=1)
    inception_3b_3_3_reduce = conv2d(inception_3a_concat, weights['w_inception_3b_33_reduce'], biases['b_inception_3b_33_reduce'], strides=1)
    inception_3b_3_3 = conv2d(inception_3b_3_3_reduce, weights['w_inception_3b_33'], biases['b_inception_3b_33'], strides=1)
    inception_3b_5_5_reduce = conv2d(inception_3a_concat, weights['w_inception_3b_55_reduce'], biases['b_inception_3b_55_reduce'], strides=1)
    inception_3b_5_5 = conv2d(inception_3b_5_5_reduce, weights['w_inception_3b_55'], biases['b_inception_3b_55'], strides=1)
    inception_3b_maxpool = maxpool2d(inception_3a_concat, k=3, strides=1)
    inception_3b_maxpool_reduce = conv2d(inception_3b_maxpool, weights['w_inception_3b_mp_reduce'], biases['b_inception_3b_mp_reduce'], strides=1)

    inception_3b_concat = tf.concat(3, [inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_maxpool_reduce])

    pool3_3_3 = maxpool2d(inception_3b_concat, k=3, strides=2)

    # Inception module (4a)
    inception_4a_1_1 = conv2d(pool3_3_3, weights['w_inception_4a_11'], biases['b_inception_4a_11'], strides=1)
    inception_4a_3_3_reduce = conv2d(pool3_3_3, weights['w_inception_4a_33_reduce'], biases['b_inception_4a_33_reduce'], strides=1)
    inception_4a_3_3 = conv2d(inception_4a_3_3_reduce, weights['w_inception_4a_33'], biases['b_inception_4a_33'], strides=1)
    inception_4a_5_5_reduce = conv2d(pool3_3_3, weights['w_inception_4a_55_reduce'], biases['b_inception_4a_55_reduce'], strides=1)
    inception_4a_5_5 = conv2d(inception_4a_5_5_reduce, weights['w_inception_4a_55'], biases['b_inception_4a_55'], strides=1)
    inception_4a_maxpool = maxpool2d(pool3_3_3, k=3, strides=1)
    inception_4a_maxpool_reduce = conv2d(inception_4a_maxpool, weights['w_inception_4a_mp_reduce'], biases['b_inception_4a_mp_reduce'], strides=1)

    inception_4a_concat = tf.concat(3, [inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_maxpool_reduce])

    # Inception module (4b)
    inception_4b_1_1 = conv2d(inception_4a_concat, weights['w_inception_4b_11'], biases['b_inception_4b_11'], strides=1)
    inception_4b_3_3_reduce = conv2d(inception_4a_concat, weights['w_inception_4b_33_reduce'], biases['b_inception_4b_33_reduce'], strides=1)
    inception_4b_3_3 = conv2d(inception_4b_3_3_reduce, weights['w_inception_4b_33'], biases['b_inception_4b_33'], strides=1)
    inception_4b_5_5_reduce = conv2d(inception_4a_concat, weights['w_inception_4b_55_reduce'], biases['b_inception_4b_55_reduce'], strides=1)
    inception_4b_5_5 = conv2d(inception_4b_5_5_reduce, weights['w_inception_4b_55'], biases['b_inception_4b_55'], strides=1)
    inception_4b_maxpool = maxpool2d(inception_4a_concat, k=3, strides=1)
    inception_4b_maxpool_reduce = conv2d(inception_4b_maxpool, weights['w_inception_4b_mp_reduce'], biases['b_inception_4b_mp_reduce'], strides=1)

    inception_4b_concat = tf.concat(3, [inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_maxpool_reduce])

    # Inception module (4c)
    inception_4c_1_1 = conv2d(inception_4b_concat, weights['w_inception_4c_11'], biases['b_inception_4c_11'], strides=1)
    inception_4c_3_3_reduce = conv2d(inception_4b_concat, weights['w_inception_4c_33_reduce'], biases['b_inception_4c_33_reduce'], strides=1)
    inception_4c_3_3 = conv2d(inception_4c_3_3_reduce, weights['w_inception_4c_33'], biases['b_inception_4c_33'], strides=1)
    inception_4c_5_5_reduce = conv2d(inception_4b_concat, weights['w_inception_4c_55_reduce'], biases['b_inception_4c_55_reduce'], strides=1)
    inception_4c_5_5 = conv2d(inception_4c_5_5_reduce, weights['w_inception_4c_55'], biases['b_inception_4c_55'], strides=1)
    inception_4c_maxpool = maxpool2d(inception_4b_concat, k=3, strides=1)
    inception_4c_maxpool_reduce = conv2d(inception_4c_maxpool, weights['w_inception_4c_mp_reduce'], biases['b_inception_4c_mp_reduce'], strides=1)

    inception_4c_concat = tf.concat(3, [inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_maxpool_reduce])

    # Inception module (4d)
    inception_4d_1_1 = conv2d(inception_4c_concat, weights['w_inception_4d_11'], biases['b_inception_4d_11'], strides=1)
    inception_4d_3_3_reduce = conv2d(inception_4c_concat, weights['w_inception_4d_33_reduce'], biases['b_inception_4d_33_reduce'], strides=1)
    inception_4d_3_3 = conv2d(inception_4d_3_3_reduce, weights['w_inception_4d_33'], biases['b_inception_4d_33'], strides=1)
    inception_4d_5_5_reduce = conv2d(inception_4c_concat, weights['w_inception_4d_55_reduce'], biases['b_inception_4d_55_reduce'], strides=1)
    inception_4d_5_5 = conv2d(inception_4d_5_5_reduce, weights['w_inception_4d_55'], biases['b_inception_4d_55'], strides=1)
    inception_4d_maxpool = maxpool2d(inception_4c_concat, k=3, strides=1)
    inception_4d_maxpool_reduce = conv2d(inception_4d_maxpool, weights['w_inception_4d_mp_reduce'], biases['b_inception_4d_mp_reduce'], strides=1)

    inception_4d_concat = tf.concat(3, [inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_maxpool_reduce])

    # Inception module (4e)
    inception_4e_1_1 = conv2d(inception_4d_concat, weights['w_inception_4e_11'], biases['b_inception_4e_11'], strides=1)
    inception_4e_3_3_reduce = conv2d(inception_4d_concat, weights['w_inception_4e_33_reduce'], biases['b_inception_4e_33_reduce'], strides=1)
    inception_4e_3_3 = conv2d(inception_4e_3_3_reduce, weights['w_inception_4e_33'], biases['b_inception_4e_33'], strides=1)
    inception_4e_5_5_reduce = conv2d(inception_4d_concat, weights['w_inception_4e_55_reduce'], biases['b_inception_4e_55_reduce'], strides=1)
    inception_4e_5_5 = conv2d(inception_4e_5_5_reduce, weights['w_inception_4e_55'], biases['b_inception_4e_55'], strides=1)
    inception_4e_maxpool = maxpool2d(inception_4d_concat, k=3, strides=1)
    inception_4e_maxpool_reduce = conv2d(inception_4e_maxpool, weights['w_inception_4e_mp_reduce'], biases['b_inception_4e_mp_reduce'], strides=1)

    inception_4e_concat = tf.concat(3, [inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_maxpool_reduce])

    pool4_3_3 = maxpool2d(inception_4e_concat, k=3, strides=2)

    # Inception module (5a)
    inception_5a_1_1 = conv2d(pool4_3_3, weights['w_inception_5a_11'], biases['b_inception_5a_11'], strides=1)
    inception_5a_3_3_reduce = conv2d(pool4_3_3, weights['w_inception_5a_33_reduce'], biases['b_inception_5a_33_reduce'],
                                     strides=1)
    inception_5a_3_3 = conv2d(inception_5a_3_3_reduce, weights['w_inception_5a_33'], biases['b_inception_5a_33'],
                              strides=1)
    inception_5a_5_5_reduce = conv2d(pool4_3_3, weights['w_inception_5a_55_reduce'], biases['b_inception_5a_55_reduce'],
                                     strides=1)
    inception_5a_5_5 = conv2d(inception_5a_5_5_reduce, weights['w_inception_5a_55'], biases['b_inception_5a_55'],
                              strides=1)
    inception_5a_maxpool = maxpool2d(pool4_3_3, k=3, strides=1)
    inception_5a_maxpool_reduce = conv2d(inception_5a_maxpool, weights['w_inception_5a_mp_reduce'],
                                         biases['b_inception_5a_mp_reduce'], strides=1)

    inception_5a_concat = tf.concat(3,
                                    [inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_maxpool_reduce])

    # Inception module (5b)
    inception_5b_1_1 = conv2d(inception_5a_concat, weights['w_inception_5b_11'], biases['b_inception_5b_11'], strides=1)
    inception_5b_3_3_reduce = conv2d(inception_5a_concat, weights['w_inception_5b_33_reduce'],
                                     biases['b_inception_5b_33_reduce'], strides=1)
    inception_5b_3_3 = conv2d(inception_5b_3_3_reduce, weights['w_inception_5b_33'], biases['b_inception_5b_33'],
                              strides=1)
    inception_5b_5_5_reduce = conv2d(inception_5a_concat, weights['w_inception_5b_55_reduce'],
                                     biases['b_inception_5b_55_reduce'], strides=1)
    inception_5b_5_5 = conv2d(inception_5b_5_5_reduce, weights['w_inception_5b_55'], biases['b_inception_5b_55'],
                              strides=1)
    inception_5b_maxpool = maxpool2d(inception_5a_concat, k=3, strides=1)
    inception_5b_maxpool_reduce = conv2d(inception_5b_maxpool, weights['w_inception_5b_mp_reduce'],
                                         biases['b_inception_5b_mp_reduce'], strides=1)

    inception_5b_concat = tf.concat(3,
                                    [inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_maxpool_reduce])

    pool5_8_8 = avgpool2d(inception_5b_concat, k=8, strides=8)

    pool5_8_8_dropout = tf.nn.dropout(pool5_8_8, keep_prob)

    fc = tf.reshape(pool5_8_8_dropout, [-1, weights['w_fc'].get_shape().as_list()[0]])

    out = tf.add(tf.matmul(fc, weights['w_fc']), biases['b_fc'])

    return out

weights = {
    # 7x7 conv, 1 input, 64 outputs
    'w_c1_77': tf.Variable(tf.random_normal([7, 7, 3, 64], stddev=0.01)),
    # 1x1 conv, 64 input, 64 outputs
    'w_c2_11': tf.Variable(tf.random_normal([1, 1, 64, 64], stddev=0.01)),
    # 3x3 conv, 64 input, 192 outputs
    'w_c2_33': tf.Variable(tf.random_normal([3, 3, 64, 192], stddev=0.01)),

    # Inception module (3a)
    # 1x1 conv, 192 input, 64 outputs
    'w_inception_3a_11': tf.Variable(tf.random_normal([1, 1, 192, 64], stddev=0.01)),
    # 1x1 conv, 192 input, 96 outputs
    'w_inception_3a_33_reduce': tf.Variable(tf.random_normal([1, 1, 192, 96], stddev=0.01)),
    # 3x3 conv, 96 input, 128 outputs
    'w_inception_3a_33': tf.Variable(tf.random_normal([3, 3, 96, 128], stddev=0.01)),
    # 1x1 conv, 192 input, 16 outputs
    'w_inception_3a_55_reduce': tf.Variable(tf.random_normal([1, 1, 192, 16], stddev=0.01)),
    # 5x5 conv, 16 input, 32 outputs
    'w_inception_3a_55': tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.01)),
    # 1x1 conv, 192 input, 32 outputs
    'w_inception_3a_mp_reduce': tf.Variable(tf.random_normal([1, 1, 192, 32], stddev=0.01)),

    # Inception module (3b)
    # 1x1 conv, 256 input, 128 outputs
    'w_inception_3b_11': tf.Variable(tf.random_normal([1, 1, 256, 128], stddev=0.01)),
    # 1x1 conv, 256 input, 128 outputs
    'w_inception_3b_33_reduce': tf.Variable(tf.random_normal([1, 1, 256, 128], stddev=0.01)),
    # 3x3 conv, 128 input, 192 outputs
    'w_inception_3b_33': tf.Variable(tf.random_normal([3, 3, 128, 192], stddev=0.01)),
    # 1x1 conv, 256 input, 32 outputs
    'w_inception_3b_55_reduce': tf.Variable(tf.random_normal([1, 1, 256, 32], stddev=0.01)),
    # 5x5 conv, 32 input, 96 outputs
    'w_inception_3b_55': tf.Variable(tf.random_normal([5, 5, 32, 96], stddev=0.01)),
    # 1x1 conv, 192 input, 32 outputs
    'w_inception_3b_mp_reduce': tf.Variable(tf.random_normal([1, 1, 256, 64], stddev=0.01)),

    # Inception module (4a)
    # 1x1 conv, 480 input, 192 outputs
    'w_inception_4a_11': tf.Variable(tf.random_normal([1, 1, 480, 192], stddev=0.01)),
    # 1x1 conv, 480 input, 96 outputs
    'w_inception_4a_33_reduce':  tf.Variable(tf.random_normal([1, 1, 480, 96], stddev=0.01)),
    # 3x3 conv, 96 input, 208 outputs
    'w_inception_4a_33': tf.Variable(tf.random_normal([3, 3, 96, 208], stddev=0.01)),
    # 1x1 conv, 480 input, 16 outputs
    'w_inception_4a_55_reduce': tf.Variable(tf.random_normal([1, 1, 480, 16], stddev=0.01)),
    # 5x5 conv, 16 input, 48 outputs
    'w_inception_4a_55': tf.Variable(tf.random_normal([5, 5, 16, 48], stddev=0.01)),
    # 1x1 conv, 480 input, 64 outputs
    'w_inception_4a_mp_reduce': tf.Variable(tf.random_normal([1, 1, 480, 64], stddev=0.01)),

    # Inception module (4b)
    # 1x1 conv, 512 input, 160 outputs
    'w_inception_4b_11': tf.Variable(tf.random_normal([1, 1, 512, 160], stddev=0.01)),
    # 1x1 conv, 512 input, 112 outputs
    'w_inception_4b_33_reduce': tf.Variable(tf.random_normal([1, 1, 512, 112], stddev=0.01)),
    # 3x3 conv, 112 input, 224 outputs
    'w_inception_4b_33': tf.Variable(tf.random_normal([3, 3, 112, 224], stddev=0.01)),
    # 1x1 conv, 512 input, 24 outputs
    'w_inception_4b_55_reduce': tf.Variable(tf.random_normal([1, 1, 512, 24], stddev=0.01)),
    # 5x5 conv, 24 input, 64 outputs
    'w_inception_4b_55': tf.Variable(tf.random_normal([5, 5, 24, 64], stddev=0.01)),
    # 1x1 conv, 512 input, 64 outputs
    'w_inception_4b_mp_reduce': tf.Variable(tf.random_normal([1, 1, 512, 64], stddev=0.01)),

    # Inception module (4c)
    # 1x1 conv, 512 input, 128 outputs
    'w_inception_4c_11': tf.Variable(tf.random_normal([1, 1, 512, 128], stddev=0.01)),
    # 1x1 conv, 512 input, 128 outputs
    'w_inception_4c_33_reduce': tf.Variable(tf.random_normal([1, 1, 512, 128], stddev=0.01)),
    # 3x3 conv, 128 input, 256 outputs
    'w_inception_4c_33': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01)),
    # 1x1 conv, 512 input, 24 outputs
    'w_inception_4c_55_reduce': tf.Variable(tf.random_normal([1, 1, 512, 24], stddev=0.01)),
    # 5x5 conv, 24 input, 64 outputs
    'w_inception_4c_55': tf.Variable(tf.random_normal([5, 5, 24, 64], stddev=0.01)),
    # 1x1 conv, 512 input, 64 outputs
    'w_inception_4c_mp_reduce': tf.Variable(tf.random_normal([1, 1, 512, 64], stddev=0.01)),

    # Inception module (4d)
    # 1x1 conv, 512 input, 112 outputs
    'w_inception_4d_11': tf.Variable(tf.random_normal([1, 1, 512, 112], stddev=0.01)),
    # 1x1 conv, 512 input, 144 outputs
    'w_inception_4d_33_reduce': tf.Variable(tf.random_normal([1, 1, 512, 144], stddev=0.01)),
    # 3x3 conv, 144 input, 288 outputs
    'w_inception_4d_33': tf.Variable(tf.random_normal([3, 3, 144, 288], stddev=0.01)),
    # 1x1 conv, 512 input, 32 outputs
    'w_inception_4d_55_reduce': tf.Variable(tf.random_normal([1, 1, 512, 32], stddev=0.01)),
    # 5x5 conv, 32 input, 64 outputs
    'w_inception_4d_55': tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01)),
    # 1x1 conv, 512 input, 64 outputs
    'w_inception_4d_mp_reduce': tf.Variable(tf.random_normal([1, 1, 512, 64], stddev=0.01)),

    # Inception module (4e)
    # 1x1 conv, 528 input, 256 outputs
    'w_inception_4e_11': tf.Variable(tf.random_normal([1, 1, 528, 256], stddev=0.01)),
    # 1x1 conv, 528 input, 160 outputs
    'w_inception_4e_33_reduce': tf.Variable(tf.random_normal([1, 1, 528, 160], stddev=0.01)),
    # 3x3 conv, 160 input, 320 outputs
    'w_inception_4e_33': tf.Variable(tf.random_normal([3, 3, 160, 320], stddev=0.01)),
    # 1x1 conv, 528 input, 32 outputs
    'w_inception_4e_55_reduce': tf.Variable(tf.random_normal([1, 1, 528, 32], stddev=0.01)),
    # 5x5 conv, 32 input, 128 outputs
    'w_inception_4e_55': tf.Variable(tf.random_normal([5, 5, 32, 128], stddev=0.01)),
    # 1x1 conv, 528 input, 128 outputs
    'w_inception_4e_mp_reduce': tf.Variable(tf.random_normal([1, 1, 528, 128], stddev=0.01)),

    # Inception module (5a)
    # 1x1 conv, 832 input, 256 outputs
    'w_inception_5a_11': tf.Variable(tf.random_normal([1, 1, 832, 256], stddev=0.01)),
    # 1x1 conv, 832 input, 160 outputs
    'w_inception_5a_33_reduce': tf.Variable(tf.random_normal([1, 1, 832, 160], stddev=0.01)),
    # 3x3 conv, 160 input, 320 outputs
    'w_inception_5a_33': tf.Variable(tf.random_normal([3, 3, 160, 320], stddev=0.01)),
    # 1x1 conv, 832 input, 32 outputs
    'w_inception_5a_55_reduce': tf.Variable(tf.random_normal([1, 1, 832, 32], stddev=0.01)),
    # 5x5 conv, 32 input, 128 outputs
    'w_inception_5a_55': tf.Variable(tf.random_normal([5, 5, 32, 128], stddev=0.01)),
    # 1x1 conv, 832 input, 128 outputs
    'w_inception_5a_mp_reduce': tf.Variable(tf.random_normal([1, 1, 832, 128], stddev=0.01)),

    # Inception module (5b)
    # 1x1 conv, 832 input, 384 outputs
    'w_inception_5b_11': tf.Variable(tf.random_normal([1, 1, 832, 384], stddev=0.01)),
    # 1x1 conv, 832 input, 192 outputs
    'w_inception_5b_33_reduce': tf.Variable(tf.random_normal([1, 1, 832, 192], stddev=0.01)),
    # 3x3 conv, 192input, 384 outputs
    'w_inception_5b_33': tf.Variable(tf.random_normal([3, 3, 192, 384], stddev=0.01)),
    # 1x1 conv, 832 input, 48 outputs
    'w_inception_5b_55_reduce': tf.Variable(tf.random_normal([1, 1, 832, 48], stddev=0.01)),
    # 5x5 conv, 48 input, 128 outputs
    'w_inception_5b_55': tf.Variable(tf.random_normal([5, 5, 48, 128], stddev=0.01)),
    # 1x1 conv, 832 input, 128 outputs
    'w_inception_5b_mp_reduce': tf.Variable(tf.random_normal([1, 1, 832, 128], stddev=0.01)),

    # Fully-Connected
    'w_fc': tf.Variable(tf.random_normal([1024, 17], stddev=0.01))
}


biases = {
    'b_c1_77': tf.Variable(tf.random_normal([64], stddev=0.01)),
    'b_c2_11': tf.Variable(tf.random_normal([64], stddev=0.01)),
    'b_c2_33': tf.Variable(tf.random_normal([192], stddev=0.01)),

    'b_inception_3a_11': tf.Variable(tf.random_normal([64], stddev=0.01)),
    'b_inception_3a_33_reduce': tf.Variable(tf.random_normal([96], stddev=0.01)),
    'b_inception_3a_33': tf.Variable(tf.random_normal([128], stddev=0.01)),
    'b_inception_3a_55_reduce': tf.Variable(tf.random_normal([16], stddev=0.01)),
    'b_inception_3a_55': tf.Variable(tf.random_normal([32], stddev=0.01)),
    'b_inception_3a_mp_reduce': tf.Variable(tf.random_normal([32], stddev=0.01)),

    'b_inception_3b_11': tf.Variable(tf.random_normal([128], stddev=0.01)),
    'b_inception_3b_33_reduce': tf.Variable(tf.random_normal([128], stddev=0.01)),
    'b_inception_3b_33': tf.Variable(tf.random_normal([192], stddev=0.01)),
    'b_inception_3b_55_reduce': tf.Variable(tf.random_normal([32], stddev=0.01)),
    'b_inception_3b_55': tf.Variable(tf.random_normal([96], stddev=0.01)),
    'b_inception_3b_mp_reduce': tf.Variable(tf.random_normal([64], stddev=0.01)),

    'b_inception_4a_11': tf.Variable(tf.random_normal([192], stddev=0.01)),
    'b_inception_4a_33_reduce': tf.Variable(tf.random_normal([96], stddev=0.01)),
    'b_inception_4a_33': tf.Variable(tf.random_normal([208], stddev=0.01)),
    'b_inception_4a_55_reduce': tf.Variable(tf.random_normal([16], stddev=0.01)),
    'b_inception_4a_55':  tf.Variable(tf.random_normal([48], stddev=0.01)),
    'b_inception_4a_mp_reduce': tf.Variable(tf.random_normal([64], stddev=0.01)),

    'b_inception_4b_11': tf.Variable(tf.random_normal([160], stddev=0.01)),
    'b_inception_4b_33_reduce': tf.Variable(tf.random_normal([112], stddev=0.01)),
    'b_inception_4b_33': tf.Variable(tf.random_normal([224], stddev=0.01)),
    'b_inception_4b_55_reduce': tf.Variable(tf.random_normal([24], stddev=0.01)),
    'b_inception_4b_55': tf.Variable(tf.random_normal([64], stddev=0.01)),
    'b_inception_4b_mp_reduce': tf.Variable(tf.random_normal([64], stddev=0.01)),

    'b_inception_4c_11': tf.Variable(tf.random_normal([128], stddev=0.01)),
    'b_inception_4c_33_reduce': tf.Variable(tf.random_normal([128], stddev=0.01)),
    'b_inception_4c_33': tf.Variable(tf.random_normal([256], stddev=0.01)),
    'b_inception_4c_55_reduce': tf.Variable(tf.random_normal([24], stddev=0.01)),
    'b_inception_4c_55': tf.Variable(tf.random_normal([64], stddev=0.01)),
    'b_inception_4c_mp_reduce': tf.Variable(tf.random_normal([64], stddev=0.01)),

    'b_inception_4d_11': tf.Variable(tf.random_normal([112], stddev=0.01)),
    'b_inception_4d_33_reduce': tf.Variable(tf.random_normal([144], stddev=0.01)),
    'b_inception_4d_33': tf.Variable(tf.random_normal([288], stddev=0.01)),
    'b_inception_4d_55_reduce': tf.Variable(tf.random_normal([32], stddev=0.01)),
    'b_inception_4d_55': tf.Variable(tf.random_normal([64], stddev=0.01)),
    'b_inception_4d_mp_reduce': tf.Variable(tf.random_normal([64], stddev=0.01)),

    'b_inception_4e_11': tf.Variable(tf.random_normal([256], stddev=0.01)),
    'b_inception_4e_33_reduce': tf.Variable(tf.random_normal([160], stddev=0.01)),
    'b_inception_4e_33': tf.Variable(tf.random_normal([320], stddev=0.01)),
    'b_inception_4e_55_reduce': tf.Variable(tf.random_normal([32], stddev=0.01)),
    'b_inception_4e_55': tf.Variable(tf.random_normal([128], stddev=0.01)),
    'b_inception_4e_mp_reduce': tf.Variable(tf.random_normal([128], stddev=0.01)),

    'b_inception_5a_11': tf.Variable(tf.random_normal([256], stddev=0.01)),
    'b_inception_5a_33_reduce': tf.Variable(tf.random_normal([160], stddev=0.01)),
    'b_inception_5a_33': tf.Variable(tf.random_normal([320], stddev=0.01)),
    'b_inception_5a_55_reduce': tf.Variable(tf.random_normal([32], stddev=0.01)),
    'b_inception_5a_55': tf.Variable(tf.random_normal([128], stddev=0.01)),
    'b_inception_5a_mp_reduce': tf.Variable(tf.random_normal([128], stddev=0.01)),

    'b_inception_5b_11': tf.Variable(tf.random_normal([384], stddev=0.01)),
    'b_inception_5b_33_reduce': tf.Variable(tf.random_normal([192], stddev=0.01)),
    'b_inception_5b_33': tf.Variable(tf.random_normal([384], stddev=0.01)),
    'b_inception_5b_55_reduce': tf.Variable(tf.random_normal([48], stddev=0.01)),
    'b_inception_5b_55': tf.Variable(tf.random_normal([128], stddev=0.01)),
    'b_inception_5b_mp_reduce': tf.Variable(tf.random_normal([128], stddev=0.01)),

    'b_fc':  tf.Variable(tf.random_normal([17], stddev=0.01))
}

X = tf.placeholder(tf.float32, [None, 227, 227, 3])
Y = tf.placeholder(tf.float32, [None, 17])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

op = model(X, weights, biases, keep_prob)

#### Network design is finished.

x, y = oxflower17.load_data(one_hot=True)
trX, trY, teX, teY = x[0:1224], y[0:1224], x[1224:1360], y[1224:1360]
trX = trX.reshape(-1, 227, 227, 3)
teX = teX.reshape(-1, 227, 227, 3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(op, Y))
train_op = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost)
#train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

predict_op = tf.equal(tf.argmax(op, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(predict_op, tf.float32))

# Launch the graph in a session
# This code is extracted from "http://pythonkim.tistory.com/56"
# Some variables are changed
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(1000):

        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], keep_prob: 4.0})

        print(i, sess.run(accuracy, feed_dict={X: teX[test_indices], Y: teY[test_indices], keep_prob: 1.0}))

