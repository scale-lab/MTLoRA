# This code is referenced from
# https://github.com/facebookresearch/astmt/
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# License: Attribution-NonCommercial 4.0 International

import numpy as np
import torch
from evaluation.eval_sal_no_beta import SaliencyMeterWithNoBeta
from evaluation.eval_sal_beta import SaliencyMeterWithBeta

import evaluation.jaccard as evaluation


class SaliencyMeter(object):
    def __init__(self, ignore_index=255, threshold_step=0.05, beta_squared=0.3):
        self.no_beta = SaliencyMeterWithNoBeta()
        self.with_beta = SaliencyMeterWithBeta(
            ignore_index=ignore_index, threshold_step=threshold_step, beta_squared=beta_squared)

    @torch.no_grad()
    def update(self, pred, gt):
        self.no_beta.update(pred, gt)
        self.with_beta.update(pred, gt)

    def reset(self):
        self.no_beta.reset()
        self.with_beta.reset()

    def get_score(self, verbose=True):
        no_beta_result = self.no_beta.get_score(verbose=False)
        with_beta_result = self.with_beta.get_score(verbose=False)
        eval_result = {
            'Beta maxF': with_beta_result['maxF'],
            'maxF': no_beta_result['maxF'],
            'mIoU': no_beta_result['mIoU'],
        }

        if verbose:
            print('\nResults for Saliency Estimation')
            print('Beta maxF: {:.3f}'.format(100.0 * with_beta_result['maxF']))
            print('maxF: {:.3f}'.format(100.0 * no_beta_result['maxF']))
            print('mIoU: {:.3f}'.format(100.0 * no_beta_result['mIoU']))

        return eval_result
