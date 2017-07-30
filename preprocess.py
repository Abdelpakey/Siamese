
# coding: utf-8

# In[1]:

import os
import xml.etree.ElementTree as ET
import struct
from scipy import misc
import numpy as np
import ntpath
import cv2
import pdb
import time

current_time = lambda: int(round(time.time() * 1000))


# In[ ]:




# In[2]:

# video_ids.sh


# In[3]:

root_dir = 'ILSVRC'
root_crop_dir = 'ILSVRC_crop'


# In[4]:

root_data_dir = os.path.join(root_dir, 'Data/VID/train')


# In[5]:

dirs = os.listdir(root_data_dir)
dirs.sort()


# In[6]:

video_idx = 0
records = []
for dir in dirs:
    dir_path = os.path.join(root_data_dir, dir)
    sub_dirs = os.listdir(dir_path)
    sub_dirs.sort()
    for sub_dir in sub_dirs:
        if "txt" in sub_dir:
            continue
        sub_dir_path = os.path.join(dir_path, sub_dir)
        file_num = len(os.listdir(sub_dir_path))
        record = sub_dir_path + " " + str(video_idx) + " " + str(file_num)
        video_idx += 1
        records.append(record)


# In[7]:

with open("vid_id_frames.txt", 'w') as f:
    f.write('\n'.join(records))


# In[ ]:




# In[ ]:




# In[8]:

# per_frame_annotation.m and parse_objects.ml
# Read per-frame XML annotations and write bbox and track info on txt files
# Reads per_frame bbox and track information and generates per-video reports


# In[9]:

class_names = ['n02691156','n02419796','n02131653','n02834778','n01503061','n02924116','n02958343','n02402425',
               'n02084071','n02121808','n02503517','n02118333','n02510455','n02342885','n02374451','n02129165',
               'n01674464','n02484322','n03790512','n02324045','n02509815','n02411705','n01726692','n02355227',
               'n02129604','n04468005','n01662784','n04530566','n02062744','n02391049']


# In[10]:

root_anno_dir = os.path.join(root_dir, 'Annotations/VID/train')


# In[11]:

dirs = os.listdir(root_anno_dir)
dirs.sort()


# In[17]:

for i, dir in enumerate(dirs):
    dir_path = os.path.join(root_anno_dir, dir)
    sub_dirs = os.listdir(dir_path)
    sub_dirs.sort()
    data_dir_path = os.path.join(root_data_dir, dir)
    for j, sub_dir in enumerate(sub_dirs):
        sub_dir_path = os.path.join(dir_path, sub_dir)
        xml_files = os.listdir(sub_dir_path)
        xml_files.sort()
        data_sub_dir_path = os.path.join(data_dir_path, sub_dir)
        vid_records = []
        for k, xml_file in enumerate(xml_files):
            xml_path = os.path.join(sub_dir_path, xml_file)
            img_path = os.path.join(data_sub_dir_path, xml_file).replace("xml", "JPEG")
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size = root.find("size")
            width = size.find("width").text
            height = size.find("height").text
            
            objects = root.findall("object")
            records = []
            for o_id, obj in enumerate(objects):
                class_name = obj.find('name').text
                class_id = class_names.index(class_name)
                track_id = obj.find('trackid').text
                xmax = obj.find('bndbox').find('xmax').text
                xmin = obj.find('bndbox').find('xmin').text
                ymax = obj.find('bndbox').find('ymax').text
                ymin = obj.find('bndbox').find('ymin').text
                
                box_height = int(ymax) - int(ymin) + 1
                box_width = int(xmax) - int(xmin) + 1
                
#                 record = str(i) + "," + str(j) + "," + str(k) + "," + str(o_id) + "," + track_id + "," + \
#                          str(class_id) + "," + width + "," + height + "," + xmin + "," + ymin + "," + \
#                          str(box_width) + "," + str(box_height) + "." + img_path
#                 records.append(record)
                
                vid_record = track_id + "," + str(class_id) + "," + width + "," + height + "," + xmin + "," +                              ymin + "," + str(box_width) + "," + str(box_height) + "," + img_path
                vid_records.append(vid_record)
#             img_anno_path = img_path.replace(".JPEG", ".txt")
#             with open(img_anno_path, 'w') as f:
#                 f.write('\n'.join(records))
                
        with open(data_sub_dir_path + ".txt", 'w') as f:
            f.write('\n'.join(vid_records))
    break


# In[ ]:




# In[8]:

# save_crops.m
# Extract and save crops from video


# In[9]:

exemplar_size = 127;
instance_size = 255;
context_amount = 0.5;


# In[10]:

dirs = os.listdir(root_data_dir)
dirs.sort()


# In[11]:

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


# In[12]:

def get_crops(img, xmin, ymin, box_width, box_height):
    xmax = xmin + box_width - 1
    ymax = ymin + box_height - 1
    xcenter = (xmin + xmax) / 2
    ycenter = (ymin + ymax) / 2
    
    box_width_z = box_width + context_amount * (box_width + box_height)
    box_height_z = box_height + context_amount * (box_width + box_height)
    
    s_z = np.sqrt(box_width_z * box_height_z)
    scale_z = exemplar_size / s_z
    
    img_crop_z, left_pad_z, top_pad_z, right_pad_z, bottom_pad_z =         get_subwindow_avg(img, (ycenter, xcenter), (exemplar_size, exemplar_size), (int(s_z), int(s_z)))
    
    pad_z = np.ceil([scale_z * left_pad_z,
                     scale_z * top_pad_z,
                     exemplar_size - scale_z * (right_pad_z + left_pad_z),
                     exemplar_size - scale_z * (top_pad_z + bottom_pad_z)])

    s_x = s_z * instance_size / exemplar_size
    scale_x = instance_size / s_x
    img_crop_x, left_pad_x, top_pad_x, right_pad_x, bottom_pad_x =         get_subwindow_avg(img, (ycenter, xcenter), (instance_size, instance_size), (int(s_x), int(s_x)))
    pad_x = np.ceil([scale_x * left_pad_x,
                     scale_x * top_pad_x,
                     instance_size - scale_x * (right_pad_x + left_pad_x),
                     instance_size - scale_x * (top_pad_x + bottom_pad_x)])
    
    return img_crop_z, pad_z, img_crop_x, pad_x


# In[15]:

root_crop_data_dir = os.path.join(root_crop_dir, 'Data/VID/train')
for i, dir in enumerate(dirs):
    print(dir)
    dir_path = os.path.join(root_data_dir, dir)
    crop_dir_path = os.path.join(root_crop_data_dir, dir)
    sub_dirs = os.listdir(dir_path)
    sub_dirs.sort()
    for j, sub_dir in enumerate(sub_dirs):
        sub_dir_path = os.path.join(dir_path, sub_dir)
        if sub_dir_path.endswith("txt") == False:
            continue
        print("    " + sub_dir)
        crop_sub_dir_path = os.path.join(crop_dir_path, sub_dir.split('.')[0])
        with open(sub_dir_path, 'r') as f:
            records = f.readlines()
        for record in records:
            record = record.strip()
            info = record.split(',')
            track_id, class_id, width, height, xmin, ymin, box_width, box_height, img_path =                 [int(info[0]), int(info[1]), int(info[2]), int(info[3]), int(info[4]), int(info[5]), int(info[6]),                  int(info[7]), info[8]]
            img = misc.imread(img_path)
            
            img_name = ntpath.basename(img_path).split('.')[0]
            crop_img_z_name = img_name + "_" + "%02d" % track_id + ".crop.z.jpg"
            crop_img_x_name = img_name + "_" + "%02d" % track_id + ".crop.x.jpg"
            
            crop_img_z_path = os.path.join(crop_sub_dir_path, crop_img_z_name)
            crop_img_x_path = os.path.join(crop_sub_dir_path, crop_img_x_name)
            
#             if os.path.isfile(crop_img_z_path) and os.path.isfile(crop_img_x_path):
#                 continue

            img_crop_z, pad_z, img_crop_x, pad_x = get_crops(img, xmin, ymin, box_width, box_height)
            
            misc.imsave(crop_img_z_path, img_crop_z)
            misc.imsave(crop_img_x_path, img_crop_x)


# In[ ]:



