# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
#
# Original file:
# License: Attribution-NonCommercial 4.0 International (https://github.com/facebookresearch/astmt/)
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Modifications:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)

import numpy as np


def jaccard(gt, pred, void_pixels=None):

    assert (gt.shape == pred.shape)

    if void_pixels is None:
        void_pixels = np.zeros_like(gt)
    assert (void_pixels.shape == gt.shape)

    gt = gt.astype(bool)
    pred = pred.astype(bool)
    void_pixels = void_pixels.astype(bool)
    if np.isclose(np.sum(gt & np.logical_not(void_pixels)), 0) and np.isclose(np.sum(pred & np.logical_not(void_pixels)), 0):
        return 1

    else:
        return np.sum(((gt & pred) & np.logical_not(void_pixels))) / \
            np.sum(((gt | pred) & np.logical_not(void_pixels)), dtype=float)


def precision_recall(gt, pred, void_pixels=None):

    if void_pixels is None:
        void_pixels = np.zeros_like(gt)

    gt = gt.astype(bool)
    pred = pred.astype(bool)
    void_pixels = void_pixels.astype(bool)

    tp = ((pred & gt) & ~void_pixels).sum()
    fn = ((~pred & gt) & ~void_pixels).sum()

    fp = ((pred & ~gt) & ~void_pixels).sum()

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)

    return prec, rec
