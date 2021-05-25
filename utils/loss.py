import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class cross_entropy(nn.Module):
    def __init__(self):
        super(cross_entropy, self).__init__()

    def forward(self,res_prob, gt_prob):
        assert res_prob.size() == gt_prob.size() and len(res_prob) == 1
        loss = -torch.sum(gt_prob * torch.log(res_prob) + (1 - gt_prob) * torch.log(1 - res_prob))
        return loss
