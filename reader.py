import os, sys, shutil
import pickle
import numpy as np
import random
from scipy import misc
import six
from six.moves import urllib, range
import copy
import logging
import cv2
import json

from tensorpack import *

from cfgs.config import cfg

import pdb

class Data(RNGDataFlow):
    def __init__(self, data_size=None, frame_range=50):
        # read the vid_id_frames
        with open('vid_id_frames.txt') as f:
            videos = f.readlines()
        self.videos = [e.split(' ')[0] for e in videos]
        self.data_size = len(videos) if data_size == None else data_size
        self.frame_range = frame_range


        # generate labels and label_weights
        self.labels = np.zeros((cfg.score_size, cfg.score_size))
        self.label_weights = np.zeros((cfg.score_size, cfg.score_size))

        ct = cfg.score_size // 2

        def dist(p0, p1):
            return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

        pos_num = 0
        neg_num = 0
        for i in range(cfg.score_size):
            for j in range(cfg.score_size):
                if dist((i,j), (ct,ct)) <= cfg.pos_radius:
                    self.labels[i,j] = 1
                    pos_num += 1
                else:
                    self.labels[i,j] = -1
                    neg_num += 1

        for i in range(cfg.score_size):
            for j in range(cfg.score_size):
                if self.labels[i,j] == 1:
                    self.label_weights[i,j] = 1 / pos_num
                else:
                    self.label_weights[i,j] = 1 / neg_num

        self.labels = np.expand_dims(self.labels, axis=3)
        self.label_weights = np.expand_dims(self.label_weights, axis=3)

    def size(self):
        return self.data_size

    def generate_sample(self):
        # randomly choose a video
        vid = random.choice(self.videos[0:400])
        # vid = 'ILSVRC/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00010001'
        vid_text = vid + ".txt"
        with open(vid_text) as f:
            records = f.readlines()
        records = [e.strip().split(',') for e in records]

        # randomly choose a track id
        track_ids = [e[0] for e in records]
        track_id = random.choice(list(set(track_ids)))

        # get all objects with that track id
        track_objs = [e for e in records if e[0] == track_id]

        obj_z = random.choice(track_objs)
        idx_z = track_objs.index(obj_z)

        possible_x_start = np.max([0, idx_z - self.frame_range])
        possible_x_end = np.min([len(track_objs), idx_z + self.frame_range])
        possible_x_objs = track_objs[possible_x_start:idx_z] + track_objs[idx_z:possible_x_end]

        obj_x = random.choice(possible_x_objs)

        ori_z_path = obj_z[-1].replace("ILSVRC/", "ILSVRC_crop/").replace(".JPEG", "")
        ori_x_path = obj_x[-1].replace("ILSVRC/", "ILSVRC_crop/").replace(".JPEG", "")

        crop_z_path = ori_z_path + "_" + "%02d" % int(track_id) + ".crop.z.jpg"
        crop_x_path = ori_x_path + "_" + "%02d" % int(track_id) + ".crop.x.jpg"

        crop_z_img = misc.imread(crop_z_path)
        crop_x_img = misc.imread(crop_x_path)


        return [crop_z_img, crop_x_img, self.labels, self.label_weights]

    def get_data(self):
        for k in range(self.size()):
            yield self.generate_sample()

if __name__ == '__main__':
    ds = Data()
    ds.reset_state()
    dp_producer = ds.get_data()
    dp = next(dp_producer)
    pdb.set_trace()
