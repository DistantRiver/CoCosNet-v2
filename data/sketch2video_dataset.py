# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
import torch
import numpy as np
import math
import random
from PIL import Image

from data.pix2pix_dataset import Pix2pixDataset
from data.base_dataset import get_params, get_transform


class Sketch2VideoDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(no_pairing_check=True)
        parser.set_defaults(load_size=275)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(label_nc=20)
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
            label_paths.append(name)
            image_path = name.replace('sketch', 'video').rsplit('.', 1)[0] + '.jpg'
            image_paths.append(os.path.join(image_path))
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
            old_key = items[0]
            key = self.labelpath_to_imgpath(old_key)
            if old_key in self_pair_dict.keys():
                val = [self.labelpath_to_imgpath(it) for it in self_pair_dict[items[0]]]
            else:
                val = [self.labelpath_to_imgpath(items[-1])]
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
        label_path = path.replace('video', 'sketch', 1).rsplit('.', 1)[0] + '.png'
        return label_path

    def labelpath_to_imgpath(self, path):
        img_path = path.replace('sketch', 'video', 1).rsplit('.', 1)[0] + '.jpg'
        return img_path
    
    def get_label_tensor(self, path):
        label = Image.open(path).convert('L')
        params1 = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params1)
        label_tensor = transform_label(label)
        # label_tensor[label_tensor == 255] = self.opt.label_nc
        # 'unknown' is opt.label_nc
        
        input_semantics = torch.zeros(61, self.opt.crop_size, self.opt.crop_size)
        label_dir = path.replace('sketch', 'sketch_segmentation', 1).rsplit('.', 1)[0]
        dir_name = label_dir.rsplit(os.sep, 1)[1]
        for i in range(61):
            cls_name = dir_name + '_' + str(i) + '.png'
            cls_path = os.path.join(label_dir, cls_name)
            if os.path.exists(cls_path):
                cls_file = Image.open(cls_path).convert('L')
                cls_tensor = transform_label(cls_file)
                input_semantics[i] = cls_tensor
        
        label_tensor = torch.cat((label_tensor, input_semantics), dim=0)
        
        return label_tensor, params1
