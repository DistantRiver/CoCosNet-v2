# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import math
import random
from PIL import Image
import pickle
import torchvision.transforms as transforms

from data.pix2pix_dataset import Pix2pixDataset
from data.base_dataset import get_params, get_transform


class Video2SketchFlowDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(no_pairing_check=True)
        parser.set_defaults(load_size=275)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(label_nc=185)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        if opt.phase == 'train':
            with open('./data/sketchvideo_train.txt', 'r') as fd:
                lines = [line.strip() for line in fd.readlines() if line.strip()]
        elif opt.phase == 'test':
            with open('./data/sketchvideo_val.txt', 'r') as fd:
                lines = [line.strip() for line in fd.readlines() if line.strip()]
        image_paths = []
        label_paths = []
        for i in range(len(lines)):
            name = lines[i]
            image_paths.append(name)
            label_path = name.replace('sketch', 'video').rsplit('.', 1)[0] + '.jpg'
            label_paths.append(os.path.join(label_path))
        return label_paths, image_paths

    def get_ref_video_like(self, opt):
        pair_path = './data/sketchvideo_self_pair.txt'
        with open(pair_path) as fd:
            self_pair = fd.readlines()
            self_pair = [it.strip() for it in self_pair]
        self_pair_dict = {}
        for it in self_pair:
            items = it.split(',')
            self_pair_dict[items[0]] = items[1:]
        ref_path = './data/sketchvideo_ref_test.txt' if opt.phase == 'test' else './data/sketchvideo_ref.txt'
        with open(ref_path) as fd:
            ref = fd.readlines()
            ref = [it.strip() for it in ref]
        ref_dict = {}
        for i in range(len(ref)):
            items = ref[i].strip().split(',')
            key = items[0]
            if key in self_pair_dict.keys():
                val = [it for it in self_pair_dict[items[0]]]
            else:
                val = [items[-1]]
            ref_dict[key.replace('\\',"/")] = [v.replace('\\',"/") for v in val]
        train_test_folder = ('', '')
        return ref_dict, train_test_folder

    def get_ref_vgg(self, opt):
        extra = ''
        if opt.phase == 'test':
            extra = '_test'
        with open('./data/sketchvideo_ref{}.txt'.format(extra)) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            if opt.phase == 'test':
                val = [it for it in items[1:]]
            else:
                val = [items[-1]]
            ref_dict[key.replace('\\',"/")] = [v.replace('\\',"/") for v in val]
        train_test_folder = ('', '')
        return ref_dict, train_test_folder

    def get_ref(self, opt):
        if opt.video_like:
            return self.get_ref_video_like(opt)
        else:
            return self.get_ref_vgg(opt)

    def imgpath_to_labelpath(self, path):
        label_path = path.replace('sketch', 'video').rsplit('.', 1)[0] + '.jpg'
        return label_path

    def labelpath_to_imgpath(self, path):
        img_path = path.replace('video', 'sketch').rsplit('.', 1)[0] + '.png'
        return img_path
    
    def get_label_tensor(self, path, id1, id2):
        label = Image.open(path).convert('RGB')
        params1 = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params1)
        label_tensor = transform_label(label)
        # label_tensor[label_tensor == 255] = self.opt.label_nc
        # 'unknown' is opt.label_nc
        
        if id1 == id2:
            flow_tensor = torch.zeros(2, self.opt.crop_size, self.opt.crop_size)
            label_tensor = torch.cat((label_tensor, flow_tensor), dim=0)
            return label_tensor, params1
        
        flow_dir = '/home/ubuntu/datasets/SketchVideo/video/flow'
        video_name = path.rsplit(os.sep, 1)[1].rsplit('-', 1)[0]
        flow_path = os.path.join(flow_dir, video_name + '.pkl')
        with open(flow_path, 'rb') as sp:
            flow_data = pickle.load(sp)
        frame_num, _, h, w = flow_data.shape
        indexes = [0]
        step = frame_num / 4
        start = 0
        for i in range(3):
            start += step
            indexes.append(int(start))
        indexes.append(frame_num - 1)
        
        index_1 = indexes[id1 - 1]
        index_2 = indexes[id2 - 1]
        if id1 < id2:
            flow_cut = flow_data[index_1:index_2]
        else:
            flow_cut = -flow_data[index_2:index_1]
        flow_tensor = flow_cut.sum(0)
        
        flow_tensor = torch.from_numpy(flow_tensor)
        flow_transform = transforms.Compose([transforms.Normalize((0.5), (0.5))])
        flow_tensor = flow_transform(flow_tensor)
        
        if w > h:
            pad_width_1 = (w - h) // 2
            pad_width_2 = w - h - pad_width_1
            flow_tensor = F.pad(flow_tensor, (0,0,pad_width_1, pad_width_2,0,0), value=0.0)
        else:
            pad_width_1 = (h - w) // 2
            pad_width_2 = h - w - pad_width_1
            flow_tensor = F.pad(flow_tensor, (pad_width_1, pad_width_2,0,0,0,0), value=0.0)
        
        flow_tensor = F.interpolate(flow_tensor.unsqueeze(0), size=[self.opt.load_size,self.opt.load_size], mode='nearest').squeeze()
        x, y = params1['crop_pos']
        flow_tensor = flow_tensor[:, y:(y+self.opt.crop_size), x:(x+self.opt.crop_size)]
        
        label_tensor = torch.cat((label_tensor, flow_tensor), dim=0)
        
        return label_tensor, params1
