import tensorflow as tf
import os
from data import *


def configure_networks(imgs, img_z, img_w, img_h, keep_prob):
    ## Weight Initialization
    # Create lots of weights and biases & Initialize with a small positive number as we will use ReLU
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    ## Convolution and Pooling
    # Convolution here: stride=1, zero-padded -> output size = input size
    def conv3d(x, W):
        return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')  # conv2d, [1, 1, 1, 1]

    # Pooling: max pooling over 2x2 blocks
    def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
        return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')

    ## First Convolutional Layer
    # Conv then Max-pooling. 1st layer will have 32 features for each 5x5 patch. (1 feature -> 32 features)
    with tf.variable_scope('LeNet'):
        W_conv1 = weight_variable([5, 5, 5, 1, 32])  # shape of weight tensor = [5,5,1,32]
        b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]

        # Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
        x_image = tf.reshape(imgs, [-1, img_z, img_w, img_h, 1])  # [-1,28,28,1]
        print(x_image.get_shape)  # (?, 256, 256, 40, 1)  # -> output image: 28x28 x1

        # x_image * weight tensor + bias -> apply ReLU -> apply max-pool
        h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)  # conv2d, ReLU(x_image * weight + bias)
        print(h_conv1.get_shape)  # (?, 256, 256, 40, 32)  # -> output image: 28x28 x32
        h_pool1 = max_pool_2x2(h_conv1)  # apply max-pool
        print(h_pool1.get_shape)  # (?, 128, 128, 20, 32)  # -> output image: 14x14 x32

        ## Second Convolutional Layer
        # Conv then Max-pooling. 2nd layer will have 64 features for each 5x5 patch. (32 features -> 64 features)
        W_conv2 = weight_variable([5, 5, 5, 32, 64])  # [5, 5, 32, 64]
        b_conv2 = bias_variable([64])  # [64]

        h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)  # conv2d, .ReLU(x_image * weight + bias)
        print(h_conv2.get_shape)  # (?, 128, 128, 20, 64)  # -> output image: 14x14 x64
        h_pool2 = max_pool_2x2(h_conv2)  # apply max-pool
        print(h_pool2.get_shape)  # (?, 64, 64, 10, 64)    # -> output image: 7x7 x64

        ## Densely Connected Layer (or fully-connected layer)
        # fully-connected layer with 1024 neurons to process on the entire image
        W_fc1 = weight_variable([3 * 2 * 2 * 64, 1024])  # [7*7*64, 1024]
        b_fc1 = bias_variable([1024])  # [1024]]

        h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 2 * 2 * 64])  # -> output image: [-1, 7*7*64] = 3136
        print(h_pool2_flat.get_shape)  # (?, 2621440)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
        print(h_fc1.get_shape)  # (?, 1024)  # -> output: 1024

        ## Dropout (to reduce overfitting; useful when training very large neural network)
        # We will turn on dropout during training & turn off during testing

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        print(h_fc1_drop.get_shape)  # -> output: 1024

        ## Readout Layer
        W_fc2 = weight_variable([1024, 2])  # [1024, 10]
        b_fc2 = bias_variable([2])  # [10]

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        print(y_conv.get_shape)
        return y_conv


def train(conf):
    if not os.path.exists(conf.modeldir):
        os.makedirs(conf.modeldir)
    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    if conf.balance_data:
        train_datas, test_datas, img_z, img_w, img_h = load_balance_data(conf.train_size)
        train_imgs = [train_datas[i]['img'] for i in range(len(train_datas))]
        train_labels = [train_datas[i]['label'] for i in range(len(train_datas))]
        test_imgs = [test_datas[i]['img'] for i in range(len(test_datas))]
        test_labels = [test_datas[i]['label'] for i in range(len(test_datas))]
    else:
        train_imgs, train_labels, test_imgs, test_labels, img_z, img_w, img_h = load_data(conf.train_size)
    total_size = len(train_labels) + len(test_labels)

    x = tf.placeholder(tf.float32, [None, img_z, img_w, img_h], name='x-input')
    y = tf.placeholder(tf.int32, name='y-input')
    keep_prob = tf.placeholder(tf.float32)
    one_hot_y = tf.one_hot(y, conf.class_num)
    pred = configure_networks(x, img_z, img_w, img_h, keep_prob)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(conf.epochs):
            x_train, y_train = train_imgs, train_labels
            total_train_accuracy = 0
            total_train_loss = 0
            train_output = []
            for offset in range(0, len(y_train), conf.batch_size):
                end = offset + conf.batch_size
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                _, trainacc, output, loss = sess.run([training_operation, accuracy_operation, pred, loss_operation],
                                               feed_dict={x: batch_x, y: batch_y, keep_prob: conf.keep_r})
                train_output = train_output + np.argmax(output, 1).tolist()
                total_train_accuracy += (trainacc * len(batch_x))
                total_train_loss += loss
            train_accuracy = total_train_accuracy / len(y_train)
            train_loss = total_train_loss / len(y_train)
            
            x_test, y_test = test_imgs, test_labels
            total_test_accuracy = 0
            total_test_loss = 0
            test_output = []
            for offset in range(0, len(y_test), conf.batch_size):
                end = offset + conf.batch_size
                batch_x, batch_y = x_test[offset:end], y_test[offset:end]
                testacc, output, loss = sess.run([accuracy_operation, pred, loss_operation],
                                           feed_dict={x: batch_x, y: batch_y, keep_prob: conf.keep_r})
                test_output = test_output + np.argmax(output, 1).tolist()
                total_test_loss += loss
                total_test_accuracy += (testacc * len(batch_x))
            test_accuracy = total_test_accuracy / len(y_test)
            test_loss = total_test_loss / len(y_test)
            
            print('Train\nTrue', train_output, '\nPred', train_labels)
            print('Test\nTrue', test_output, '\nPred', test_labels)
            print('epoch %4d train_loss %.4f train_acc %.4f test_acc %.4f test_loss %.4f' % (i + 1, train_loss, train_accuracy, test_accuracy, test_loss))
        saver.save(sess, './modeldir')
    print("Model saved")
