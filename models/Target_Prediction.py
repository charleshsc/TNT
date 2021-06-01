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
        :param x: (bs, N, 64 + 2)
        :return: (bs, N, 1)
        '''
        x = self.fc(x)
        #x = F.relu(F.layer_norm(x,x.size()))
        x = F.relu(F.layer_norm(x,x.size()[-1:]))
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
        :param target_point: target coordinates (bs,N,2)
        :param x:  scene context feature (bs,N, 64)
        :return: \pi: (bs, N, ), v_x: (bs, N,2)
        '''
        assert len(target_point.size()) == len(x.size()) and target_point.size()[1] == x.size()[1]
        input_x = torch.cat([target_point,x],dim=-1) # (bs, N, 66)
        f_x = self.f(input_x).squeeze(2) # (bs, N,)
        out_pi = F.log_softmax(f_x,dim=-1) # (bs, N)
        v_x = self.v(input_x) # (bs, N, 2)
        return out_pi, v_x

    def get_sort_idx(self, out_pi):
        # out_pi (bs, N)
        tmp = out_pi.cpu().detach().numpy()
        tmp = np.argsort(-tmp) # (bs, N)
        tmp_list = []
        for i in range(len(tmp)):
            tmp_list.append(tmp[i][:self.M][np.newaxis,:]) # (1, M)
        return np.concatenate(tmp_list,0) # (bs, M)

    def forward_M(self, target_point, x):
        '''
        :param target_point: target coordinates (bs, N,2)
        :param x: scene context feature (bs, N, 64)
        :return: \pi: (bs, M, ), v_x: (bs, M,2), sort_idx (bs, M, )
        '''
        out_pi, v_x = self(target_point,x)
        out_pi = torch.exp(out_pi) # (bs, N)
        sort_idx = self.get_sort_idx(out_pi) #(bs, M)
        return sort_idx, np.array([out_pi[bs][sort_idx[bs]] for bs in range(len(out_pi))]), np.array([v_x[bs][sort_idx[bs]] for bs in range(len(v_x))])

    def _loss(self,target_point, x, u, delta_xy, idx):
        '''
        :param u:  target closest to the ground truth location (bs, 2,)
        :param delta_xy: the spatial offsets of u from the ground truth (bs, 2, )
        :param idx: the index for the u in target_point (bs,)
        :return: loss
        '''
        out_pi, v_x = self(target_point, x) # (bs, N) , (bs,N,2)
        loss1 = self.L_cls(out_pi, idx)

        target_v_xy = v_x.gather(dim=1,index=idx.repeat(2,1).T.unsqueeze(1)).squeeze(1) # (bs, 2)
        assert target_v_xy.size() == delta_xy.size()
        loss2 = self.L_offset(target_v_xy, delta_xy)
        return loss1 + loss2



