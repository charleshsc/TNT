import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Cross_Entropy import cross_entropy

class MLP(nn.Module):
    def __init__(self, input_channels=102, hidden_channels=64, output_channels=1):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels,output_channels)

    def forward(self, x):
        '''
        :param x: (bs, M, 64+2*T)
        :return: (bs, M, 1)
        '''
        x = self.fc(x)
        x = F.relu(F.layer_norm(x,x.size()[-1:]))
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
        :param s_F: (bs, M, T, 2)
        :param x: (bs, M, 64 )
        :return: (bs, M )
        '''
        assert s_F.size()[1] == x.size()[1]
        input_x = torch.cat([s_F.flatten(start_dim=2),x],dim=-1) # (bs, M, 64+2*T)
        g_x = self.g(input_x).squeeze(2)
        phi_x = F.softmax(g_x,dim=-1)
        return phi_x

    def _loss(self, s_F, x, s_gt):
        '''
        :param s_F: (bs, M, T, 2)
        :param x: (bs, M, 64)
        :param s_gt: (bs, T, 2)
        :return:
        '''
        M = s_F.size()[1]

        def D(s, s_GT):
            '''
            :param s: (bs,M,T,2)
            :param s_GT: (bs,M,T,2)
            :return: (bs,M)
            '''
            assert s.size() == s_GT.size(), "s size:"+str(s.size()) + "s_GT size: "+str(s_GT.size())
            distance = torch.sum(torch.square(s-s_GT),dim=-1)
            return torch.max(distance,dim=-1).values

        with torch.no_grad():
            s_m_gt = []
            for gt in s_gt:
                s_m_gt.append(gt.repeat(M,1).reshape(M,gt.size()[0],gt.size()[1]).unsqueeze(0)) # (1, M, T, 2)
            s_m_gt = torch.cat(s_m_gt, 0) # (bs, M ,T, 2)
            psi = D(s_F,s_m_gt) # (bs,M)
            psi = -psi / self.alpha
            psi = F.softmax(psi, dim=-1)  # label (bs, M)

        phi = self(s_F, x) #(bs, M)
        assert phi.size() == psi.size() and phi.size()[1] == self.M
        loss = self.loss_fn(phi, psi)
        return loss






