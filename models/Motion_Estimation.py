import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_channels=66, hidden_channels=64, output_channels=1):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels,output_channels)

    def forward(self, x):
        '''
        :param x: (num, 64 + 2)
        :return: (num, 1)
        '''
        x = self.fc(x)
        x = F.relu(F.layer_norm(x,x.size()[-1:]))
        x = self.fc2(x)
        return x

class Motion_Estimator(nn.Module):
    def __init__(self, input_channels=66, hidden_channels=64, output_channels=1):
        # output_channels = 2 * T
        super(Motion_Estimator, self).__init__()
        self.estimator = MLP(input_channels, hidden_channels, output_channels)

        self.L_reg = nn.SmoothL1Loss()

    def forward(self, target_point, x):
        '''
        :param target_point: (num, 2)
        :param x: (num, 64)
        :return: (num, T, 2)
        '''
        if len(target_point.size()) == 2:
            input_x = torch.cat([target_point,x],dim=-1) # (num, 66)
            estimation = self.estimator(input_x).view(x.size()[0],-1,2)
        elif len(target_point.size()) == 3:
            input_x = torch.cat([target_point, x], dim=-1)  # (bs, num, 66)
            estimation = self.estimator(input_x).view(x.size()[0], x.size()[1], -1, 2)
        else:
            raise ValueError("the Motion Estimator size if wrong")
        return estimation

    def _loss(self, target_point, x, gt, origin_point):
        '''
        :param target_point: (bs, 2 )
        :param x: (bs, 64 )
        :param gt: (bs, T, 2)
        :param origin_point: (bs, 2 )
        :return: scalar
        '''
        result = self(target_point,x) # (bs, T, 2)
        assert result.size() == gt.size(), 'result size:' + str(result.size())+' gt size:'+str(gt.size())
        result = torch.cat([origin_point.unsqueeze(1),result],dim=1)
        with torch.no_grad():
            gt = torch.cat([origin_point.unsqueeze(1),gt],dim=1)
        res_perstep_offset = []
        gt_perstep_offset = []
        for idx in range(result.size()[1]-1):
            res_perstep_offset.append((result[:,idx+1]-result[:,idx]).unsqueeze(1))
            with torch.no_grad():
                gt_perstep_offset.append((gt[:,idx+1] - gt[:,idx]).unsqueeze(1))
        res_perstep_offset = torch.cat(res_perstep_offset,dim=1) # (bs,T,2)
        gt_perstep_offset = torch.cat(gt_perstep_offset,dim=1) # (bs,T,2)
        assert res_perstep_offset.size() == gt_perstep_offset.size()
        loss = self.L_reg(res_perstep_offset,gt_perstep_offset)
        return loss
