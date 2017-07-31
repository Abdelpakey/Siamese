import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from reader import *
from cfgs.config import cfg

import pdb

class Model(ModelDesc):

    def __init__(self, batch_size=1):
        super(Model, self).__init__()
        self.batch_size = batch_size

    def _get_inputs(self):
        return [InputDesc(tf.uint8, [None, cfg.exemplar_size, cfg.exemplar_size, 3], 'exemplar_img'),
                InputDesc(tf.uint8, [None, cfg.search_size, cfg.search_size, 3], 'search_img'),
                InputDesc(tf.float32, [None, cfg.score_size, cfg.score_size, 1], 'labels'),
                InputDesc(tf.float32, [None, cfg.score_size, cfg.score_size, 1], 'label_weights')]

    def _build_graph(self, inputs):
        exemplar_img, search_img, labels, label_weights = inputs

        tf.summary.image('exemplar_img', exemplar_img)
        tf.summary.image('search_img', search_img)

        exemplar_img = tf.cast(exemplar_img, tf.float32) * (1.0 / 255)
        search_img = tf.cast(search_img, tf.float32) * (1.0 / 255)

        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        exemplar_img = (exemplar_img - image_mean) / image_std
        search_img = (search_img - image_mean) / image_std

        def network(l):
            with argscope(Conv2D, nl=tf.identity, use_bias=False, padding='VALID',
                          W_init=variance_scaling_initializer(mode='FAN_OUT')):
                l = (LinearWrap(l)
                     .Conv2D('conv0', 96, 11, stride=2, nl=BNReLU)
                     .MaxPooling('pool0', shape=3, stride=2)
                     .Conv2D('conv1', 256, 5, nl=BNReLU, split=2)
                     .MaxPooling('pool1', shape=3, stride=2)
                     .Conv2D('conv2', 192, 3, nl=BNReLU)
                     .Conv2D('conv3', 192, 3, nl=BNReLU, split=2)
                     .Conv2D('conv4', 128, 3, split=2)())
            return l

        with tf.variable_scope("siamese-fc") as scope:
            net_z = network(exemplar_img)
            scope.reuse_variables()
            net_x = network(search_img)

        # net_z and net_x are [B, H, W, C]
        net_z = tf.transpose(net_z, perm=[1, 2, 0, 3])
        net_x = tf.transpose(net_x, perm=[1, 2, 0, 3])

        # net_z and net_x are [H, W, B, C]
        Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
        Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))

        net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
        net_x = tf.reshape(net_x, (1, Hx, Wx, B*Cx))

        # net_x is [1, Hx, Wx, B*C]
        # net_z is [Hz, Wz, B*C, 1]
        net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')

        # net_final is [1, Hf, Wf, B*C]
        net_final = tf.concat(tf.split(net_final, self.batch_size, axis=3), axis=0)

        # net_final is [B, Hf, Wf, C]
        net_final = tf.expand_dims(tf.reduce_mean(net_final, axis=3), axis=3)
        # net_final = tf.expand_dims(tf.reduce_sum(net_final, axis=3), axis=3)

        # net_final is [B, Hf, Wf, 1]
        final_bias = tf.Variable(tf.zeros([]), name='final_bias')
        net_final = net_final + final_bias * tf.ones((self.batch_size, cfg.score_size, cfg.score_size, 1))

        net_final = tf.identity(net_final, name='prediction')

        loss = tf.log(1 + tf.exp(-net_final * labels)) * label_weights
        loss = tf.reduce_mean(tf.reduce_sum(loss, (1,2,3)), name='loss')

        wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        add_moving_summary(loss, wd_cost)
        self.cost = tf.add_n([loss, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 10 ** cfg.start_lr, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test, batch_size):
    isTrain = train_or_test == 'train'

    ds = Data(data_size=cfg.steps_per_epoch)

    if isTrain:
        augmentors = [
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Clip(),
            imgaug.Flip(horiz=True),
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            imgaug.ToUint8()
        ]
    # ds = AugmentImageComponent(ds, augmentors)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    ds = BatchData(ds, batch_size, remainder=not isTrain)
    return ds


def get_config(args):
    ds_train = get_data('train', args.batch_size)

    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(),
            # HyperParamSetterWithFunc('learning_rate',
            #                          lambda e, x: 10 ** (cfg.start_lr + (cfg.end_lr - cfg.start_lr) * 1.0 * e / (cfg.max_epoch - 1)) ),
            ScheduledHyperParamSetter('learning_rate',
                                     [(0, 1e-2), (50, 3e-3), (100, 1e-3), (150, 3e-4), (200, 1e-4)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(args.batch_size),
        max_epoch=cfg.max_epoch,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    args.batch_size = int(args.batch_size) // NR_GPU

    logger.auto_set_dir()
    config = get_config(args)
    if args.load:
        config.session_init = get_model_loader(args.load)
    config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()
