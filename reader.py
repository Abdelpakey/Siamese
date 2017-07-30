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

import pdb

class Data(RNGDataFlow):
    def __init__(self, data_size=None, frame_range=50):
        # read the vid_id_frames
        with open('vid_id_frames.txt') as f:
            videos = f.readlines()
        self.videos = [e.split(' ')[0] for e in videos]
        self.data_size = len(videos) if data_size == None else data_size
        self.frame_range = frame_range

    def size(self):
        return self.data_size

    def generate_sample(self):
        # randomly choose a video
        vid = random.choice(self.videos[0:130])
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

        return [crop_z_img, crop_x_img]

    def get_data(self):
        for k in range(self.size()):
            yield self.generate_sample()

if __name__ == '__main__':
    ds = Data()
    ds.reset_state()
    dp_producer = ds.get_data()
    dp = next(dp_producer)
