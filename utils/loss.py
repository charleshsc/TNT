import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class cross_entropy(nn.Module):
    def __init__(self):
        super(cross_entropy, self).__init__()

    def forward(self,res_prob, gt_prob):
        # (bs, M)
        assert res_prob.size() == gt_prob.size() and len(res_prob.size()) == 2
        loss = 0
        for single_res, single_gt in zip(res_prob, gt_prob):
            loss = loss + -torch.sum(single_gt * torch.log(single_res) + (1 - single_gt) * torch.log(1 - single_res))
        return loss/res_prob.size()[0]
