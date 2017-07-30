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

    def _get_inputs(self):
        return [InputDesc(tf.uint8, [None, cfg.exemplar_size, cfg.exemplar_size, 3], 'exemplar_img'),
                InputDesc(tf.int32, [None, cfg.search_size, cfg.search_size, 3], 'search_img')]

    def _build_graph(self, inputs):
        exemplar_img, search_img = inputs

        pdb.set_trace()

        tf.summary.image('exemplar_img', exemplar_img)
        tf.summary.image('search_img', search_img)

        image = tf.cast(image, tf.float32) * (1.0 / 255)

        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
        # image = tf.transpose(image, [0, 3, 1, 2])

        def network(l):
            pdb.set_trace()
            l = Conv2D('conv.0', l, out_channel=96, kernel_shape=(11, 11), padding='VALID', stride=2, nl=BNReLU)
            l = MaxPooling('pooling.0', l, shape=(3,3), stride=2)

            l = Conv2D('conv.1', l, out_channel=256, kernel_shape=(5, 5), padding='VALID', stride=1, nl=BNReLU, split=2)
            l = MaxPooling('pooling.0', l, shape=(3,3), stride=2)
    
            l = Conv2D('conv.2', l, out_channel=192, kernel_shape=(3, 3), padding='VALID', stride=1, nl=BNReLU)

            l = Conv2D('conv.3', l, out_channel=192, kernel_shape=(3, 3), padding='VALID', stride=1, nl=BNReLU, split=2)

            l = Conv2D('conv.4', l, out_channel=128, kernel_shape=(3, 3), padding='VALID', stride=1, split=2)

            return l


        with tf.variable_scope("siamese") as scope:
            exemplar_ebd = network(exemplar_image)
            scope.reuse_variables()
            search_ebd = network(search_image)
        

        wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        add_moving_summary(loss, wd_cost)
        self.cost = tf.add_n([loss, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'

    ds = Data()

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
    ds = AugmentImageComponent(ds, augmentors)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    return ds


def get_config():
    ds_train = get_data('train')

    return TrainConfig(
        dataflow=ds_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                                     [(0, 1e-2), (30, 3e-3), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(),
        max_epoch=160,
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
    BATCH_SIZE = int(args.batch_size) // NR_GPU

    logger.auto_set_dir()
    config = get_config()
    if args.load:
        config.session_init = get_model_loader(args.load)
    config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()
