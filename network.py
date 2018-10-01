import tensorflow as tf
import os


class cnn3d(object):
    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        self.process_data()

    def process_data(self):
        self.imgs = None
        self.labels = None
        self.img_z = None
        self.img_w, self.img_h = None, None
        # TODO:get the imgs and labels

    def configure_networks(self):
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
        W_conv1 = weight_variable([5, 5, 5, 1, 32])  # shape of weight tensor = [5,5,1,32]
        b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]

        # Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
        x_image = tf.reshape(self.imgs, [-1, self.width, self.height, self.depth, 1])  # [-1,28,28,1]
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
        W_fc1 = weight_variable([16 * 16 * 3 * 64, 1024])  # [7*7*64, 1024]
        b_fc1 = bias_variable([1024])  # [1024]]

        h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 3 * 64])  # -> output image: [-1, 7*7*64] = 3136
        print(h_pool2_flat.get_shape)  # (?, 2621440)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
        print(h_fc1.get_shape)  # (?, 1024)  # -> output: 1024

        ## Dropout (to reduce overfitting; useful when training very large neural network)
        # We will turn on dropout during training & turn off during testing
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        print(h_fc1_drop.get_shape)  # -> output: 1024

        ## Readout Layer
        W_fc2 = weight_variable([1024, self.conf.class_num])  # [1024, 10]
        b_fc2 = bias_variable([nLabel])  # [10]

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv

    def train(self):
        x = tf.placeholder(tf.float32, [self.conf.batch_size, self.img_z, self.img_w, self.img_h, 1], name='x-input')
        y = tf.placeholder(tf.float32, [None], name='y-input')
        one_hot_y = tf.one_hot(y, self.conf.class_num)
        logits = self.configure_networks()

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate)
        training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.conf.epochs):
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, self.num_samples, self.conf.batch_size):
                    end = offset + self.conf.batch_size
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    train_loss, train_accuracy = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                total_accuracy = 0
                for offset in range(0, num_examples, BATCH_SIZE):
                    batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
                    accuracy = sess.run(self.accuracy_operation, feed_dict={x: batch_x, y: batch_y})
                    total_accuracy += (accuracy * len(batch_x))
                validation_accuracy = total_accuracy / num_examples
                print("EPOCH {} ...".format(i + 1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()

            saver.save(sess, 'lenet')
            print("Model saved")

# def test(self):
#     with tf.Session() as sess:
#         saver.restore(sess, tf.train.latest_checkpoint('.'))
#
#         test_accuracy = evaluate(X_test, y_test)
#         print("Test Accuracy = {:.3f}".format(test_accuracy))
#
# def evaluate(self, X_data, y_data):
#     num_examples = len(X_data)
#     total_accuracy = 0
#     sess = tf.get_default_session()
#     for offset in range(0, num_examples, BATCH_SIZE):
#         batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
#         accuracy = sess.run(self.accuracy_operation, feed_dict={x: batch_x, y: batch_y})
#         total_accuracy += (accuracy * len(batch_x))
#     return total_accuracy / num_examples
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     # 初始化全局变量
#     tf.global_variables_initializer().run()
#     for i in range(TRAINING_STEPS):
#         xs, ys = mnist.train.next_batch(BATCH_SIZE)
#         # 断点调试：embed()
#         # 将输入的训练数据格式调整为一个四维矩阵
#         reshaped_xs = np.reshape(xs, (
#             BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
#         _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
#         # 每1000轮保存一次模型。
#         if i % 100 == 0:
#             # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。通过损失函数的大小可以大概了解训练的情况。
#             print("After %d training step(s), loss on training batch is %f." % (step, loss_value))
#             # 保存当前的模型。规定了保存的格式，global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000”表示训练1000轮后得到的模型
#             saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
#
#
# def inference(self, npy_path, test_path, model_index, train_flag=True):
# # some statistic index
# highest_acc = 0.0
# highest_iterator = 1
#
# all_filenames = get_all_filename(npy_path, self.cubic_shape[model_index][1])
# # how many time should one epoch should loop to feed all data
# times = int(len(all_filenames) / self.batch_size)
# if (len(all_filenames) % self.batch_size) != 0:
#     times = times + 1
#
# # keep_prob used for dropout
# keep_prob = tf.placeholder(tf.float32)
# # take placeholder as input
# x = tf.placeholder(tf.float32, [None, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1],
#                                 self.cubic_shape[model_index][2]])
# x_image = tf.reshape(x, [-1, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1],
#                          self.cubic_shape[model_index][2], 1])
# net_out = self.archi_1(x_image, keep_prob)
#
# saver = tf.train.Saver()  # default to save all variable,save mode or restore from path
#
# if train_flag:
#     # softmax layer
#     real_label = tf.placeholder(tf.float32, [None, 2])
#     cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label))
#     # cross_entropy = -tf.reduce_sum(real_label * tf.log(net_out))
#     net_loss = tf.reduce_mean(cross_entropy)
#
#     train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(net_loss)
#
#     correct_prediction = tf.equal(tf.argmax(net_out, 1), tf.argmax(real_label, 1))
#     accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#     merged = tf.summary.merge_all()
#
#     with tf.Session() as sess:
#         # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#         # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
#         sess.run(tf.global_variables_initializer())
#         train_writer = tf.summary.FileWriter('./tensorboard/', sess.graph)
#         # loop epoches
#         for i in range(self.epoch):
#             epoch_start = time.time()
#             #  the data will be shuffled by every epoch
#             random.shuffle(all_filenames)
#             for t in range(times):
#                 batch_files = all_filenames[t * self.batch_size:(t + 1) * self.batch_size]
#                 batch_data, batch_label = get_train_batch(batch_files)
#                 feed_dict = {x: batch_data, real_label: batch_label,
#                              keep_prob: self.keep_prob}
#                 _, summary = sess.run([train_step, merged], feed_dict=feed_dict)
#                 train_writer.add_summary(summary, i)
#                 saver.save(sess, './ckpt/archi-1', global_step=i + 1)
#
#             epoch_end = time.time()
#             test_batch, test_label = get_test_batch(test_path)
#             test_dict = {x: test_batch, real_label: test_label, keep_prob: self.keep_prob}
#             acc_test, loss = sess.run([accruacy, net_loss], feed_dict=test_dict)
#             print('accuracy  is %f' % acc_test)
#             print("loss is ", loss)
#             print(" epoch %d time consumed %f seconds" % (i, (epoch_end - epoch_start)))
#
#     print("training finshed..highest accuracy is %f,the iterator is %d " % (highest_acc, highest_iterator))
