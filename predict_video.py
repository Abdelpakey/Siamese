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


scales = np.asarray([1.04**-1, 1, 1.04])
scale_penalty = 0.97
w_influence  = 0.176
score_size = 17
response_up = 16
total_stride = 8
exemplar_size = 127
instance_size = 255
scale_lr = 0.59

def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    model = Model(scales.shape[0])
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["exemplar_img", "search_img"],
                                   output_names=["prediction"])

    predict_func = OfflinePredictor(predict_config) 
    return predict_func

# def predict_video(video_path, label_path, predict_func):
def predict_video(args, predict_func):

    cap = cv2.VideoCapture(args.video_path)
    ret, img = cap.read()
    if ret == False:
        return

    frame_idx = 0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape

    f = open(args.label_path, 'r')
    content = f.read().strip()
    xmin, ymin, box_width, box_height = [int(e) for e in content.split(',')]

    target_position = np.asarray([ymin + box_height // 2, xmin + box_width // 2])

    target_size = np.asarray([box_height, box_width])
    wc_z = target_size[1] + 0.5 * np.sum(target_size)
    hc_z = target_size[0] + 0.5 * np.sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)

    scale_z = exemplar_size / s_z
    # crop z from first frame (img_rgb)
    z_crop, _, _, _, _ = get_subwindow_avg(img_rgb,
                                           target_position,
                                           (exemplar_size, exemplar_size),
                                           [np.round(s_z).astype(int), np.round(s_z).astype(int)])

    d_search = (instance_size - exemplar_size) / 2
    pad = int(d_search / scale_z)
    s_x = s_z + 2 * pad
    min_s_x = 0.2 * s_x
    max_s_x = 5 * s_x

    video_out = cv2.VideoWriter("output.mp4",
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                25,
                                (width, height))
    img = cv2.rectangle(img,
                       (xmin, ymin),
                       (xmin + box_width, ymin + box_height),
                       (0, 0, 255),
                       3)
    video_out.write(img)
    cv2.imwrite("result_imgs/" + str(frame_idx) + ".jpg", img)

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(frame_idx)
        scaled_instance = s_x * scales
        scaled_target = np.dot(np.expand_dims(target_size, 1), np.expand_dims(scales, 0))

        x_crops = make_scale_pyramid(frame_rgb, target_position, scaled_instance, instance_size)
        x_crops = np.asarray(x_crops)

        z_crops = np.tile(np.expand_dims(z_crop, 0), (len(scales), 1, 1, 1))

        if frame_idx == 209:
            pdb.set_trace()

        predictions = predict_func([z_crops, x_crops])[0]

        # choose the best scale
        if scales.shape[0] > 1:
            cur_scale_id = int(np.floor(scales.shape[0] / 2))
            best_peak = None
            best_scale = None
            best_score = None
            for i in range(scales.shape[0]):
                score_up = cv2.resize(predictions[i], (score_size * response_up, score_size * response_up))
                if i != cur_scale_id:
                    score_up = score_up * scale_penalty
                cur_peak = np.max(score_up)
                if best_peak is None or cur_peak > best_peak:
                    best_scale = i
                    best_peak = cur_peak
                    best_score = score_up
        else:
            best_scale = 0
            best_score = cv2.resize(predictions[0], (score_size * response_up, score_size * response_up))

        best_score = best_score - np.min(best_score)
        best_score = best_score / np.sum(best_score)

        # apply windowing
        window = np.dot(np.expand_dims(np.hanning(score_size * response_up), 1), np.expand_dims(np.hanning(score_size * response_up), 0))
        window = window / np.sum(window)
        best_score = (1 - w_influence) * best_score + w_influence * window

        max_coord = np.where(best_score == np.max(best_score))
        max_coord = [e[0] for e in max_coord]

        disp = max_coord - np.floor(score_size * response_up / 2)
        disp = disp * total_stride / response_up
        disp = disp * s_x / instance_size;

        target_position = target_position + disp

        # scale damping and saturation
        s_x = max(min_s_x, min(max_s_x, (1 - scale_lr) * s_x + scale_lr * scaled_instance[best_scale]));
        target_size = (1 - scale_lr) * target_size + scale_lr * scaled_target[:, best_scale];

        xmin = int(np.round(target_position[1] - target_size[1] / 2))
        ymin = int(np.round(target_position[0] - target_size[0] / 2))
        xmax = int(np.round(xmin + target_size[1]))
        ymax = int(np.round(ymin + target_size[0]))

        img = cv2.rectangle(frame,
                           (xmin, ymin),
                           (xmax, ymax),
                           (0, 0, 255),
                           3)
        video_out.write(img)

        cv2.imwrite("result_imgs/" + str(frame_idx) + ".jpg", img)

    video_out.release()

def get_subwindow_avg(img, pos, model_sz, original_sz=None):
    avg_chans = np.mean(img, (0,1))

    if original_sz == None:
        original_sz = model_sz;

    sz = original_sz;
    
    img_shape = img.shape[0:2]
    
    c = [(sz[0] - 1) / 2, (sz[1] - 1) / 2]

    context_xmin = int(pos[1] - c[1])
    context_xmax = context_xmin + sz[1] - 1
    context_ymin = int(pos[0] - c[0])
    context_ymax = context_ymin + sz[0] - 1
    
    left_pad = np.max([0, -context_xmin])
    top_pad = np.max([0, -context_ymin])
    right_pad = np.max([0, context_xmax - img_shape[1] + 1])
    bottom_pad = np.max([0, context_ymax - img_shape[0] + 1])
    
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad
    
    pad_height = img_shape[0] + top_pad + bottom_pad
    pad_width = img_shape[1] + left_pad + right_pad
    pad_img = np.zeros((img_shape[0] + top_pad + bottom_pad, img_shape[1] + left_pad + right_pad, 3))

    avg_chans = np.expand_dims(np.expand_dims(avg_chans, 0), 0)
    pad_img = cv2.resize(avg_chans, (img_shape[1] + left_pad + right_pad, img_shape[0] + top_pad + bottom_pad))
    pad_img[top_pad:top_pad + img_shape[0], left_pad:left_pad + img_shape[1], :] = img

    img_patch_original = pad_img[context_ymin:context_ymax, context_xmin:context_xmax,:]
    
    img_patch = cv2.resize(img_patch_original, model_sz)
    
    return img_patch, left_pad, top_pad, right_pad, bottom_pad

def make_scale_pyramid(img, target_position, in_side_scaled, out_side):
    in_side_scaled = np.round(in_side_scaled).astype(int)
    max_target_side = in_side_scaled[-1]
    min_target_side = in_side_scaled[0]
    beta = out_side / min_target_side
    search_side = np.round(beta * max_target_side).astype(int)

    search_region, _, _, _, _ = get_subwindow_avg(img,
                                                  target_position,
                                                  (search_side, search_side),
                                                  [np.round(max_target_side), np.round(max_target_side)])

    pyramid = []
    for i in range(len(in_side_scaled)):
        target_side = np.round(beta * in_side_scaled[i]).astype(int)
        x_crop, _, _, _, _ = get_subwindow_avg(search_region,
                                               (1 + search_side * np.asarray([1, 1])) / 2,
                                               (out_side, out_side),
                                               [target_side, target_side])
        pyramid.append(x_crop)

    return pyramid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    parser.add_argument('--video_path', help='path of the video')
    parser.add_argument('--label_path', help='path of the label file')
    args = parser.parse_args()

    predict_func = get_pred_func(args)

    predict_video(args, predict_func)
    # predict_video(args, None)
