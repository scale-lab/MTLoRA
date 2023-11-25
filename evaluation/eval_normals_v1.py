# This code is referenced from
# https://github.com/facebookresearch/astmt/
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# License: Attribution-NonCommercial 4.0 International

import numpy as np
import math
import torch


def normal_ize(arr):
    arr_norm = np.linalg.norm(arr, ord=2, axis=2)[..., np.newaxis] + 1e-12
    return arr / arr_norm


class NormalsMeterV1(object):
    def __init__(self):
        self.eval_dict = {'mean': 0., 'rmse': 0.,
                          '11.25': 0., '22.5': 0., '30': 0., 'n': 0}

    @torch.no_grad()
    def update(self, pred, gt):
        # Performance measurement happens in pixel wise fashion (Same as code from ASTMT (above))
        pred = 2 * pred / 255 - 1
        pred = pred.permute(0, 3, 1, 2)  # [B, C, H, W]
        valid_mask = (gt != 255)
        invalid_mask = (gt == 255)

        # Put zeros where mask is invalid
        pred[invalid_mask] = 0.0
        gt[invalid_mask] = 0.0

        # Calculate difference expressed in degrees
        deg_diff_tmp = (
            180 / math.pi) * (torch.acos(torch.clamp(torch.sum(pred * gt, 1), min=-1, max=1)))
        deg_diff_tmp = torch.masked_select(deg_diff_tmp, valid_mask[:, 0])

        self.eval_dict['mean'] += torch.sum(deg_diff_tmp).item()
        self.eval_dict['rmse'] += torch.sum(
            torch.sqrt(torch.pow(deg_diff_tmp, 2))).item()
        self.eval_dict['11.25'] += torch.sum(
            (deg_diff_tmp < 11.25).float()).item() * 100
        self.eval_dict['22.5'] += torch.sum(
            (deg_diff_tmp < 22.5).float()).item() * 100
        self.eval_dict['30'] += torch.sum((deg_diff_tmp <
                                          30).float()).item() * 100
        self.eval_dict['n'] += deg_diff_tmp.numel()

    def reset(self):
        self.eval_dict = {'mean': 0., 'rmse': 0.,
                          '11.25': 0., '22.5': 0., '30': 0., 'n': 0}

    def get_score(self, verbose=True):
        eval_result = dict()
        eval_result['mean'] = self.eval_dict['mean'] / self.eval_dict['n']
        eval_result['rmse'] = self.eval_dict['mean'] / self.eval_dict['n']
        eval_result['11.25'] = self.eval_dict['11.25'] / self.eval_dict['n']
        eval_result['22.5'] = self.eval_dict['22.5'] / self.eval_dict['n']
        eval_result['30'] = self.eval_dict['30'] / self.eval_dict['n']

        if verbose:
            print('Results for Surface Normal Estimation')
            for x in eval_result:
                spaces = ""
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result
