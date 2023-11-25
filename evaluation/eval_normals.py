import torch

from evaluation.eval_normals_v1 import NormalsMeterV1
from evaluation.eval_normals_v2 import NormalsMeterV2


class NormalsMeter(object):
    def __init__(self):
        self.v1 = NormalsMeterV1()
        self.v2 = NormalsMeterV2()

    @torch.no_grad()
    def update(self, pred, gt):
        self.v1.update(pred.clone(), gt.clone())
        self.v2.update(pred, gt)

    def reset(self):
        self.v1.reset()
        self.v2.reset()

    def get_score(self, verbose=True):
        eval_v1 = self.v1.get_score(verbose=False)
        eval_v2 = self.v2.get_score(verbose=False)
        eval_result = {
            'mean': eval_v1['mean'],
            'rmse': eval_v1['rmse'],
            'mean_v2': eval_v2['mean'],
            'rmse_v2': eval_v2['rmse'],
        }

        if verbose:
            print('\nResults for Surface Normal Estimation')
            print('mean: {:.4f}'.format(eval_v1['mean']))
            print('rmse: {:.4f}'.format(eval_v1['rmse']))
            print('mean_v2: {:.4f}'.format(eval_v2['mean']))
            print('rmse_v2: {:.4f}'.format(eval_v2['rmse']))

        return eval_result
