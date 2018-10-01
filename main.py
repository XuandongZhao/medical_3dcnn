import os
import tensorflow as tf
from network import cnn3d


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('epochs', 1000, '# of step for training')
    flags.DEFINE_integer('summary_interval', 10, '# of step to save summary')
    flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
    flags.DEFINE_boolean('is_train', True, 'is train')
    flags.DEFINE_integer('class_num', 10, 'output class number')
    flags.DEFINE_integer('batch_size', 10, 'batch size')
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('test_step', 0, 'Test or predict model at this step')
    # network architecture
    flags.DEFINE_integer('ch_num', 8, 'channel number')
    flags.DEFINE_integer('layer_num', 2, 'block number')
    flags.DEFINE_float('adj_keep_r', 0.999, 'dropout keep rate')
    flags.DEFINE_float('keep_r', 0.16, 'dropout keep rate')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    # flags.DEFINE_integer('k', 8, 'top k')
    flags.DEFINE_integer('k', 4, 'top k')
    flags.DEFINE_string('first_conv', 'simple_conv', 'simple_conv, chan_conv')
    flags.DEFINE_string('second_conv', 'graph_conv', 'graph_conv, simple_conv')
    # flags.DEFINE_boolean('use_batch', True, 'use batch training')
    flags.DEFINE_boolean('use_batch', False, 'use batch training')
    flags.DEFINE_integer('batch_size', 2500, 'batch size number')
    flags.DEFINE_integer('center_num', 1500, 'start center number')
    flags.DEFINE_boolean('use_pet', True, 'use pet data')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    conf = configure()
    cnn3d(tf.Session(), conf).train()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()
