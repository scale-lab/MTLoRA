# This code is referenced from
# https://github.com/facebookresearch/astmt/
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# License: Attribution-NonCommercial 4.0 International

import torch


def normalize_tensor(input_tensor, dim):
    norm = torch.norm(input_tensor, p='fro', dim=dim, keepdim=True)
    zero_mask = (norm == 0)
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out


class NormalsMeterV2(object):
    def __init__(self, ignore_index=255):
        self.sum_deg_diff = 0
        self.total = 0
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.permute(0, 3, 1, 2)
        pred = 2 * pred / 255 - 1
        valid_mask = (gt != self.ignore_index).all(dim=1)

        pred = normalize_tensor(pred, dim=1)
        gt = normalize_tensor(gt, dim=1)
        deg_diff = torch.rad2deg(
            2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
        deg_diff = torch.masked_select(deg_diff, valid_mask)

        self.sum_deg_diff += torch.sum(deg_diff).cpu().item()
        self.total += deg_diff.numel()

    def get_score(self, verbose=False):
        eval_result = dict()
        eval_result['mean'] = self.sum_deg_diff / self.total
        eval_result['rmse'] = self.sum_deg_diff / self.total

        if verbose:
            print('Results for Surface Normal Estimation')
            print('mean: {:.3f}'.format(eval_result['mean']))
            print('rmse: {:.3f}'.format(eval_result['rmse']))

        return eval_result
