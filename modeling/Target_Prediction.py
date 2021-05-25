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
        :param x: (64 + 2,)
        :return:
        '''
        x = self.fc(x.unsqueeze(0))
        x = F.relu(F.layer_norm(x,x.size()[-1]))
        x = self.fc2(x)
        return x

class Target_predictor(nn.Module):
    def __init__(self, input_channels=66, hidden_channel=64, output_channels=1, N=1000, M=50):
        super(Target_predictor, self).__init__()
        self.f = MLP(input_channels,hidden_channel,1)
        self.v = MLP(input_channels,hidden_channel,2)
        self.N = N
        self.M = M
        self.L_cls = nn.NLLLoss()
        self.L_offset = nn.SmoothL1Loss() # Huber Loss

    def forward(self, target_point, x):
        '''
        :param target_point: target coordinates (N,2)
        :param x:  scene context feature (N, 64)
        :return: \pi: (N, ), v_x: (N,2)
        '''
        assert len(target_point.size()) == len(x.size()) and target_point.size()[0] == x.size()[0]
        input_x = torch.cat([target_point,x],dim=-1)
        f_x = self.f(input_x).squeeze() # (N,)
        out_pi = F.log_softmax(f_x,dim=-1)
        v_x = self.v(input_x) # (N, 2)
        return out_pi, v_x

    def get_sort_idx(self, out_pi):
        tmp = out_pi.detach().numpy()
        tmp = np.argsort(-tmp)
        return tmp[:self.M]

    def forward_M(self, target_point, x):
        '''
        :param target_point: target coordinates (N,2)
        :param x: scene context feature (N, 64)
        :return: \pi: (M, ), v_x: (M,2)
        '''
        out_pi, v_x = self(target_point,x)
        out_pi = torch.exp(out_pi)
        sort_idx = self.get_sort_idx(out_pi)
        return sort_idx, out_pi[sort_idx], v_x[sort_idx]

    def _loss(self,target_point, x, u, delta_xy):
        '''
        :param u:  target closest to the ground truth location (2,)
        :param delta_xy: the spatial offsets of u from the ground truth (2, )
        :return: loss
        '''
        out_pi, v_x = self(target_point, x)
        out_pi = out_pi.unsqueeze(0) # (1, N)
        target_class = None
        for idx, point in enumerate(target_point):
            if point.data.equal(u.data):
                target_class = torch.tensor([idx])
                break
        assert target_class is not None
        loss1 = self.L_cls(out_pi, target_class)
        target_v_xy = v_x[target_class]
        delta_xy = torch.tensor(delta_xy).float()
        assert target_v_xy.size() == delta_xy.size()
        loss2 = self.L_offset(target_v_xy, delta_xy)
        return loss1 + loss2




