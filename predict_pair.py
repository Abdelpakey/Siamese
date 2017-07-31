#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import ntpath
import numpy as np
from scipy import misc
import argparse
import json
import cv2

from tensorflow.python import debug as tf_debug

from tensorpack import *

from train import Model
from cfgs.config import cfg

import pdb

def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    model = Model(1)
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["exemplar_img", "search_img"],
                                   output_names=["prediction"])

    predict_func = OfflinePredictor(predict_config) 
    return predict_func

def predict_pair(img_z, img_x, predict_func):
    img_z = np.expand_dims(img_z, axis=0)
    img_x = np.expand_dims(img_x, axis=0)
    predictions = predict_func([img_z, img_x])

    pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    parser.add_argument('--img_z_path', help='path of the exemplar image')
    parser.add_argument('--img_x_path', help='path of the search image')
    args = parser.parse_args()

    img_z = misc.imread(args.img_z_path)
    img_x = misc.imread(args.img_x_path)

    predict_func = get_pred_func(args)

    predict_pair(img_z, img_x, predict_func)
