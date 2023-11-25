# This code is referenced from
# https://github.com/facebookresearch/astmt/
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# License: Attribution-NonCommercial 4.0 International

import os
import sys
import tarfile
import cv2
import re

from PIL import Image
import torch
import numpy as np
import torch.utils.data as data
import scipy.io as sio
import json

from utils import mkdir_if_missing
from torch.utils.data import DataLoader
from collections.abc import Mapping, Sequence
from torchvision import transforms
from easydict import EasyDict as edict
from skimage.morphology import thin


int_classes = int
_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


class NYUD_MT(data.Dataset):
    """
    NYUD dataset for multi-task learning.
    Includes semantic segmentation and depth prediction.

    Data can also be found at:
    https://drive.google.com/file/d/14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw/view?usp=sharing

    """

    GOOGLE_DRIVE_ID = '14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw'
    FILE = 'NYUD_MT.tgz'

    def __init__(self,
                 root,
                 split='val',
                 transform=None,
                 retname=True,
                 overfit=False,
                 do_edge=False,
                 do_semseg=False,
                 do_normals=False,
                 do_depth=False,
                 ):

        self.root = root

        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []
        _image_dir = os.path.join(root, 'images')

        # Edge Detection
        self.do_edge = do_edge
        self.edges = []
        _edge_gt_dir = os.path.join(root, 'edge')

        # Semantic segmentation
        self.do_semseg = do_semseg
        self.semsegs = []
        _semseg_gt_dir = os.path.join(root, 'segmentation')

        # Surface Normals
        self.do_normals = do_normals
        self.normals = []
        _normal_gt_dir = os.path.join(root, 'normals')

        # Depth
        self.do_depth = do_depth
        self.depths = []
        _depth_gt_dir = os.path.join(root, 'depth')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(root, 'gt_sets')

        print('Initializing dataloader for NYUD {} set'.format(''.join(self.split)))
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), 'r') as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                # Images
                _image = os.path.join(_image_dir, line + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Edges
                _edge = os.path.join(_edge_gt_dir, line + '.npy')
                assert os.path.isfile(_edge)
                self.edges.append(_edge)

                # Semantic Segmentation
                _semseg = os.path.join(_semseg_gt_dir, line + '.png')
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

                # Surface Normals
                _normal = os.path.join(_normal_gt_dir, line + '.npy')
                assert os.path.isfile(_normal)
                self.normals.append(_normal)

                # Depth Prediction
                _depth = os.path.join(_depth_gt_dir, line + '.npy')
                assert os.path.isfile(_depth)
                self.depths.append(_depth)

        if self.do_edge:
            assert (len(self.images) == len(self.edges))
        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_depth:
            assert (len(self.images) == len(self.depths))
        if self.do_normals:
            assert (len(self.images) == len(self.normals))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if self.do_edge:
            _edge = self._load_edge(index)
            if _edge.shape != _img.shape[:2]:
                _edge = cv2.resize(
                    _edge, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['edge'] = _edge

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape != _img.shape[:2]:
                print('RESHAPE SEMSEG')
                _semseg = cv2.resize(
                    _semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals(index)
            if _normals.shape[:2] != _img.shape[:2]:
                _normals = cv2.resize(
                    _normals, _img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            if _depth.shape[:2] != _img.shape[:2]:
                print('RESHAPE DEPTH')
                _depth = cv2.resize(
                    _depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert(
            'RGB')).astype(float)
        return _img

    def _load_edge(self, index):
        _edge = np.load(self.edges[index]).astype(float)
        return _edge

    def _load_semseg(self, index):
        # Note: We ignore the background class as other related works.
        _semseg = np.array(Image.open(self.semsegs[index])).astype(float)
        _semseg[_semseg == 0] = 256
        _semseg = _semseg - 1
        return _semseg

    def _load_depth(self, index):
        _depth = np.load(self.depths[index])
        return _depth

    def _load_normals(self, index):
        _normals = np.load(self.normals[index])
        return _normals

    def __str__(self):
        return 'NYUD Multitask (split=' + str(self.split) + ')'


class PASCALContext(data.Dataset):
    """
    PASCAL-Context dataset, for multiple tasks
    Included tasks:
        1. Edge detection,
        2. Semantic Segmentation,
        3. Human Part Segmentation,
        4. Surface Normal prediction (distilled),
        5. Saliency (distilled)
    """

    URL = 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL/PASCAL_MT.tgz'
    FILE = 'PASCAL_MT.tgz'

    HUMAN_PART = {1: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 1,
                      'lhand': 1, 'llarm': 1, 'llleg': 1, 'luarm': 1, 'luleg': 1, 'mouth': 1,
                      'neck': 1, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 1,
                      'rhand': 1, 'rlarm': 1, 'rlleg': 1, 'ruarm': 1, 'ruleg': 1, 'torso': 1},
                  4: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 4,
                      'lhand': 3, 'llarm': 3, 'llleg': 4, 'luarm': 3, 'luleg': 4, 'mouth': 1,
                      'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 4,
                      'rhand': 3, 'rlarm': 3, 'rlleg': 4, 'ruarm': 3, 'ruleg': 4, 'torso': 2},
                  6: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 6,
                      'lhand': 4, 'llarm': 4, 'llleg': 6, 'luarm': 3, 'luleg': 5, 'mouth': 1,
                      'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 6,
                      'rhand': 4, 'rlarm': 4, 'rlleg': 6, 'ruarm': 3, 'ruleg': 5, 'torso': 2},
                  14: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 14,
                       'lhand': 8, 'llarm': 7, 'llleg': 13, 'luarm': 6, 'luleg': 12, 'mouth': 1,
                       'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 11,
                       'rhand': 5, 'rlarm': 4, 'rlleg': 10, 'ruarm': 3, 'ruleg': 9, 'torso': 2}
                  }

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    CONTEXT_CATEGORY_LABELS = [0,
                               2, 23, 25, 31, 34,
                               45, 59, 65, 72, 98,
                               397, 113, 207, 258, 284,
                               308, 347, 368, 416, 427]

    def __init__(self,
                 root,
                 split='val',
                 transform=None,
                 area_thres=0,
                 retname=True,
                 overfit=False,
                 do_edge=True,
                 do_human_parts=False,
                 do_semseg=False,
                 do_normals=False,
                 do_sal=False,
                 num_human_parts=6,
                 ):

        self.root = root

        image_dir = os.path.join(self.root, 'JPEGImages')
        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.area_thres = area_thres
        self.retname = retname

        # Edge Detection
        self.do_edge = do_edge
        self.edges = []
        edge_gt_dir = os.path.join(self.root, 'pascal-context', 'trainval')

        # Semantic Segmentation
        self.do_semseg = do_semseg
        self.semsegs = []

        # Human Part Segmentation
        self.do_human_parts = do_human_parts
        part_gt_dir = os.path.join(self.root, 'human_parts')
        self.parts = []
        self.human_parts_category = 15
        self.cat_part = json.load(
            open(os.path.join('.', 'data/db_info/pascal_part.json'), 'r'))
        self.cat_part["15"] = self.HUMAN_PART[num_human_parts]
        self.parts_file = os.path.join(os.path.join(self.root, 'ImageSets', 'Parts'),
                                       ''.join(self.split) + '.txt')

        # Surface Normal Estimation
        self.do_normals = do_normals
        _normal_gt_dir = os.path.join(self.root, 'normals_distill')
        self.normals = []
        if self.do_normals:
            with open(os.path.join('.', 'data/db_info/nyu_classes.json')) as f:
                cls_nyu = json.load(f)
            with open(os.path.join('.', 'data/db_info/context_classes.json')) as f:
                cls_context = json.load(f)

            self.normals_valid_classes = []
            for cl_nyu in cls_nyu:
                if cl_nyu in cls_context and cl_nyu != 'unknown':
                    self.normals_valid_classes.append(cls_context[cl_nyu])

            # Custom additions due to incompatibilities
            self.normals_valid_classes.append(cls_context['tvmonitor'])

        # Saliency
        self.do_sal = do_sal
        _sal_gt_dir = os.path.join(self.root, 'sal_distill')
        self.sals = []

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(self.root, 'ImageSets', 'Context')

        self.im_ids = []
        self.images = []

        print("Initializing dataloader for PASCAL {} set".format(''.join(self.split)))
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                # Images
                _image = os.path.join(image_dir, line + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Edges
                _edge = os.path.join(edge_gt_dir, line + ".mat")
                assert os.path.isfile(_edge)
                self.edges.append(_edge)

                # Semantic Segmentation
                _semseg = self._get_semseg_fname(line)
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

                # Human Parts
                _human_part = os.path.join(part_gt_dir, line + ".mat")
                assert os.path.isfile(_human_part)
                self.parts.append(_human_part)

                _normal = os.path.join(_normal_gt_dir, line + ".png")
                assert os.path.isfile(_normal)
                self.normals.append(_normal)

                _sal = os.path.join(_sal_gt_dir, line + ".png")
                assert os.path.isfile(_sal)
                self.sals.append(_sal)

        if self.do_edge:
            assert (len(self.images) == len(self.edges))
        if self.do_human_parts:
            assert (len(self.images) == len(self.parts))
        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_normals:
            assert (len(self.images) == len(self.normals))
        if self.do_sal:
            assert (len(self.images) == len(self.sals))

        if not self._check_preprocess_parts():
            print('Pre-processing PASCAL dataset for human parts, this will take long, but will be done only once.')
            self._preprocess_parts()

        if self.do_human_parts:
            # Find images which have human parts
            self.has_human_parts = []
            for ii in range(len(self.im_ids)):
                if self.human_parts_category in self.part_obj_dict[self.im_ids[ii]]:
                    self.has_human_parts.append(1)
                else:
                    self.has_human_parts.append(0)

            # If the other tasks are disabled, select only the images that contain human parts, to allow batching
            if not self.do_edge and not self.do_semseg and not self.do_sal and not self.do_normals:
                print('Ignoring images that do not contain human parts')
                for i in range(len(self.parts) - 1, -1, -1):
                    if self.has_human_parts[i] == 0:
                        del self.im_ids[i]
                        del self.images[i]
                        del self.parts[i]
                        del self.has_human_parts[i]
            print('Number of images with human parts: {:d}'.format(
                np.sum(self.has_human_parts)))

        #  Overfit to n_of images
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]
            if self.do_edge:
                self.edges = self.edges[:n_of]
            if self.do_semseg:
                self.semsegs = self.semsegs[:n_of]
            if self.do_human_parts:
                self.parts = self.parts[:n_of]
            if self.do_normals:
                self.normals = self.normals[:n_of]
            if self.do_sal:
                self.sals = self.sals[:n_of]

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if self.do_edge:
            _edge = self._load_edge(index)
            if _edge.shape != _img.shape[:2]:
                _edge = cv2.resize(
                    _edge, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['edge'] = _edge

        if self.do_human_parts:
            _human_parts, _ = self._load_human_parts(index)
            if _human_parts.shape != _img.shape[:2]:
                _human_parts = cv2.resize(
                    _human_parts, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['human_parts'] = _human_parts

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape != _img.shape[:2]:
                _semseg = cv2.resize(
                    _semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals_distilled(index)
            if _normals.shape[:2] != _img.shape[:2]:
                _normals = cv2.resize(
                    _normals, _img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            sample['normals'] = _normals

        if self.do_sal:
            _sal = self._load_sal_distilled(index)
            if _sal.shape[:2] != _img.shape[:2]:
                _sal = cv2.resize(
                    _sal, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['sal'] = _sal

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert(
            'RGB')).astype(float)
        return _img

    def _load_edge(self, index):
        # Read Target object
        _tmp = sio.loadmat(self.edges[index])
        _edge = cv2.Laplacian(_tmp['LabelMap'], cv2.CV_64F)
        _edge = thin(np.abs(_edge) > 0).astype(float)
        return _edge

    def _load_human_parts(self, index):
        if self.has_human_parts[index]:

            # Read Target object
            _part_mat = sio.loadmat(self.parts[index])['anno'][0][0][1][0]

            _inst_mask = _target = None

            for _obj_ii in range(len(_part_mat)):

                has_human = _part_mat[_obj_ii][1][0][0] == self.human_parts_category
                has_parts = len(_part_mat[_obj_ii][3]) != 0

                if has_human and has_parts:
                    if _inst_mask is None:
                        _inst_mask = _part_mat[_obj_ii][2].astype(float)
                        _target = np.zeros(_inst_mask.shape)
                    else:
                        _inst_mask = np.maximum(
                            _inst_mask, _part_mat[_obj_ii][2].astype(float))

                    n_parts = len(_part_mat[_obj_ii][3][0])
                    for part_i in range(n_parts):
                        cat_part = str(_part_mat[_obj_ii][3][0][part_i][0][0])
                        mask_id = self.cat_part[str(
                            self.human_parts_category)][cat_part]
                        mask = _part_mat[_obj_ii][3][0][part_i][1].astype(bool)
                        _target[mask] = mask_id

            if _target is not None:
                _target, _inst_mask = _target.astype(
                    float), _inst_mask.astype(float)
            else:
                _target, _inst_mask = np.zeros((512, 512), dtype=float), np.zeros(
                    (512, 512), dtype=float)

            return _target, _inst_mask

        else:
            return np.zeros((512, 512), dtype=float), np.zeros((512, 512), dtype=float)

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index])).astype(float)

        return _semseg

    def _load_normals_distilled(self, index):
        _tmp = np.array(Image.open(self.normals[index])).astype(float)
        _tmp = 2.0 * _tmp / 255.0 - 1.0

        labels = sio.loadmat(os.path.join(
            self.root, 'pascal-context', 'trainval', self.im_ids[index] + '.mat'))
        labels = labels['LabelMap']

        _normals = np.zeros(_tmp.shape, dtype=float)

        for x in np.unique(labels):
            if x in self.normals_valid_classes:
                _normals[labels == x, :] = _tmp[labels == x, :]

        return _normals

    def _load_sal_distilled(self, index):
        _sal = np.array(Image.open(self.sals[index])).astype(float) / 255.
        _sal = (_sal > 0.5).astype(float)

        return _sal

    def _get_semseg_fname(self, fname):
        fname_voc = os.path.join(self.root, 'semseg', 'VOC12', fname + '.png')
        fname_context = os.path.join(
            self.root, 'semseg', 'pascal-context', fname + '.png')
        if os.path.isfile(fname_voc):
            seg = fname_voc
        elif os.path.isfile(fname_context):
            seg = fname_context
        else:
            seg = None
            print('Segmentation for im: {} was not found'.format(fname))

        return seg

    def _check_preprocess_parts(self):
        _obj_list_file = self.parts_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.part_obj_dict = json.load(open(_obj_list_file, 'r'))

            return list(np.sort([str(x) for x in self.part_obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess_parts(self):
        self.part_obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            if ii % 100 == 0:
                print("Processing image: {}".format(ii))
            part_mat = sio.loadmat(
                os.path.join(self.root, 'human_parts', '{}.mat'.format(self.im_ids[ii])))
            n_obj = len(part_mat['anno'][0][0][1][0])

            # Get the categories from these objects
            _cat_ids = []
            for jj in range(n_obj):
                obj_area = np.sum(part_mat['anno'][0][0][1][0][jj][2])
                obj_cat = int(part_mat['anno'][0][0][1][0][jj][1])
                if obj_area > self.area_thres:
                    _cat_ids.append(int(part_mat['anno'][0][0][1][0][jj][1]))
                else:
                    _cat_ids.append(-1)
                obj_counter += 1

            self.part_obj_dict[self.im_ids[ii]] = _cat_ids

        with open(self.parts_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(
                self.im_ids[0], json.dumps(self.part_obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(
                    ',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.part_obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing for parts finished')

    def __str__(self):
        return 'PASCAL_MT(split=' + str(self.split) + ')'


def collate_mil(batch):
    """
    Puts each data field into a tensor with outer dimension batch size.
    Custom-made for supporting MIL
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))

    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)

    elif isinstance(batch[0], str):
        return batch

    elif isinstance(batch[0], Mapping):
        batch_modified = {key: collate_mil(
            [d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        if 'edgeidx' in batch[0]:
            batch_modified['edgeidx'] = [batch[x]['edgeidx']
                                         for x in range(len(batch))]
        return batch_modified

    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate_mil(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def get_mtl_train_dataset(db_name, config, transforms):
    """ Return the train dataset """

    print('Preparing train loader for db: {}'.format(db_name))

    if db_name == 'NYUD':
        database = NYUD_MT(root=config.DATA.DATA_PATH, split='train', transform=transforms,
                           do_edge='edge' in config.TASKS,
                           do_semseg='semseg' in config.TASKS,
                           do_normals='normals' in config.TASKS,
                           do_depth='depth' in config.TASKS, overfit=False)
    elif db_name == 'PASCALContext':
        database = PASCALContext(root=config.DATA.DATA_PATH, split=['train'], transform=transforms, retname=True,
                                 do_semseg='semseg' in config.TASKS,
                                 do_edge='edge' in config.TASKS,
                                 do_normals='normals' in config.TASKS,
                                 do_sal='sal' in config.TASKS,
                                 do_human_parts='human_parts' in config.TASKS,
                                 overfit=False)
    else:
        raise NotImplemented(
            "train_db_name: Choose among PASCALContext and NYUD")

    return database


def get_tasks_config(db_name, task_list, img_size):
    """ 
        Return a dictionary with task information. 
        Additionally we return a dict with key, values to be added to the main dictionary
    """

    task_cfg = edict()
    other_args = dict()
    task_cfg.NAMES = []
    task_cfg.NUM_OUTPUT = {}
    task_cfg.FLAGVALS = {'image': cv2.INTER_CUBIC}
    task_cfg.INFER_FLAGVALS = {}

    if 'semseg' in task_list:
        # Semantic segmentation
        tmp = 'semseg'
        task_cfg.NAMES.append('semseg')
        if db_name == 'PASCALContext':
            task_cfg.NUM_OUTPUT[tmp] = 21
        elif db_name == 'NYUD':
            task_cfg.NUM_OUTPUT[tmp] = 40
        else:
            raise NotImplementedError
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_NEAREST

    if 'human_parts' in task_list:
        # Human Parts Segmentation
        assert (db_name == 'PASCALContext')
        tmp = 'human_parts'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 7
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_NEAREST

    if 'sal' in task_list:
        # Saliency Estimation
        assert (db_name == 'PASCALContext')
        tmp = 'sal'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

    if 'normals' in task_list:
        # Surface Normals
        tmp = 'normals'
        assert (db_name in ['PASCALContext', 'NYUD'])
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 3
        task_cfg.FLAGVALS[tmp] = cv2.INTER_CUBIC
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR
        other_args['normloss'] = 1  # Hard-coded L1 loss for normals

    if 'edge' in task_list:
        # Edge Detection
        assert (db_name in ['PASCALContext', 'NYUD'])
        tmp = 'edge'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR
        other_args['edge_w'] = 0.95
        other_args['eval_edge'] = False

    if 'depth' in task_list:
        # Depth
        assert (db_name == 'NYUD')
        tmp = 'depth'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR
        other_args['depthloss'] = 'l1'

    task_cfg.ALL_TASKS = edict()  # All tasks = Main tasks
    task_cfg.ALL_TASKS.NAMES = []
    task_cfg.ALL_TASKS.NUM_OUTPUT = {}
    task_cfg.ALL_TASKS.FLAGVALS = {'image': cv2.INTER_CUBIC}
    task_cfg.ALL_TASKS.INFER_FLAGVALS = {}

    for k in task_cfg.NAMES:
        task_cfg.ALL_TASKS.NAMES.append(k)
        task_cfg.ALL_TASKS.NUM_OUTPUT[k] = task_cfg.NUM_OUTPUT[k]
        task_cfg.ALL_TASKS.FLAGVALS[k] = task_cfg.FLAGVALS[k]
        task_cfg.ALL_TASKS.INFER_FLAGVALS[k] = task_cfg.INFER_FLAGVALS[k]

    task_cfg.TRAIN = {
        'SCALE': (img_size, img_size),
    }
    task_cfg.TEST = {
        'SCALE': (img_size, img_size),
    }

    return task_cfg, other_args


"""
    Transformations, datasets and dataloaders
"""


def get_transformations(db_name, config):
    """ Return transformations for training and evaluationg """
    from data import custom_transforms as tr

    # Training transformations
    if db_name == 'NYUD':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]

        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=[0], scales=[1.0, 1.2, 1.5],
                                              flagvals={x: config.ALL_TASKS.FLAGVALS[x] for x in config.ALL_TASKS.FLAGVALS})])

    elif db_name == 'PASCALContext':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]

        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25),
                                              flagvals={x: config.ALL_TASKS.FLAGVALS[x] for x in config.ALL_TASKS.FLAGVALS})])

    else:
        raise ValueError('Invalid train db name'.format(p['train_db_name']))

    # Fixed Resize to input resolution
    transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(config.TRAIN.SCALE) for x in config.ALL_TASKS.FLAGVALS},
                                         flagvals={x: config.ALL_TASKS.FLAGVALS[x] for x in config.ALL_TASKS.FLAGVALS})])
    transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_tr = transforms.Compose(transforms_tr)

    # Testing (during training transforms)
    transforms_ts = []
    transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(config.TEST.SCALE) for x in config.FLAGVALS},
                                         flagvals={x: config.FLAGVALS[x] for x in config.FLAGVALS})])
    transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_ts = transforms.Compose(transforms_ts)

    return transforms_tr, transforms_ts


def get_mtl_train_dataloader(config, dataset):
    """ Return the train dataloader """

    trainloader = DataLoader(dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, drop_last=True,
                             num_workers=config.DATA.NUM_WORKERS, collate_fn=collate_mil, pin_memory=config.DATA.PIN_MEMORY)
    return trainloader


def get_mtl_val_dataset(db_name, config, transforms):
    """ Return the validation dataset """

    print('Preparing val loader for db: {}'.format(db_name))

    if db_name == 'NYUD':
        database = NYUD_MT(root=config.DATA.DATA_PATH, split='val', transform=transforms,
                           do_edge='edge' in config.TASKS,
                           do_semseg='semseg' in config.TASKS,
                           do_normals='normals' in config.TASKS,
                           do_depth='depth' in config.TASKS, overfit=False)
    elif db_name == 'PASCALContext':
        database = PASCALContext(root=config.DATA.DATA_PATH, split=['val'], transform=transforms, retname=True,
                                 do_semseg='semseg' in config.TASKS,
                                 do_edge='edge' in config.TASKS,
                                 do_normals='normals' in config.TASKS,
                                 do_sal='sal' in config.TASKS,
                                 do_human_parts='human_parts' in config.TASKS,
                                 overfit=False)

    else:
        raise NotImplemented(
            "test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_mtl_val_dataloader(config, dataset):
    """ Return the validation dataloader """
    testloader = DataLoader(dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, drop_last=False,
                            num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY)
    return testloader
