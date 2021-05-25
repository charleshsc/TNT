import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.loss import cross_entropy

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

class Trajectory_Scorer(nn.Module):
    def __init__(self, input_channels=102, hidden_channel=64, output_channels=1,M=50, alpha=0.01):
        super(Trajectory_Scorer, self).__init__()
        self.g = MLP(input_channels, hidden_channel,output_channels)
        self.alpha = alpha
        self.M = M
        self.loss_fn = cross_entropy()

    def forward(self, s_F, x):
        '''
        :param s_F: (M, T, 2)
        :param x: (M, 64 )
        :return: (M, )
        '''
        assert s_F.size()[0] == x.size()[0]
        input_x = torch.cat([s_F.flatten(start_dim=1),x],dim=-1)
        g_x = self.g(input_x).squeeze()
        phi_x = F.softmax(g_x,dim=-1)
        return phi_x

    def _loss(self, s_F, x, s_gt):
        '''
        :param s_F: (M, T, 2)
        :param x: (M, 64)
        :param s_gt: (T, 2)
        :return:
        '''
        def D(s, s_GT):
            '''
            :param s: (T,2)
            :param s_GT: (T,2)
            :return: numpy scalar
            '''
            assert s.size() == s_GT.size()
            distance = torch.sum(torch.square(s-s_GT),dim=-1)
            return torch.max(distance).detach().numpy()

        with torch.no_grad():
            psi = []
            for s in s_F:
                psi.append(D(s, s_gt))
            psi = torch.from_numpy(np.array(psi)).squeeze()
            psi = -psi / self.alpha
            psi = F.softmax(psi, dim=-1)  # label (M, )

        phi = self(s_F, x)
        assert phi.size() == psi.size() and phi.size()[0] == self.M
        loss = self.loss_fn(phi, psi)
        return loss






