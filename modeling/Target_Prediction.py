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

