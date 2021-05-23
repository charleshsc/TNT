import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import numpy as np
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class subgraph_layer(nn.Module):
    def __init__(self, input_channels=128, hidden_channels=64):
        super(subgraph_layer, self).__init__()
        self.fc = nn.Linear(input_channels, hidden_channels) # single fully connected layer
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, inputs):
        '''
        :param input: trajectory input : (last_observe,6 | 128)
                      map input: (18 , 8 | 128)
        :return output: (last_observe | 18 , 128)
        '''
        hidden = self.fc(inputs).unsqueeze(0)
        encode_data = F.relu(F.layer_norm(hidden, hidden.size()[1:]))
        kernel_size = encode_data.size()[1]  # last_observe | 18
        maxpool = nn.MaxPool1d(kernel_size)
        polyline_feature = maxpool(encode_data.transpose(1,2)).squeeze()
        polyline_feature = polyline_feature.repeat(kernel_size, 1)
        output = torch.cat([encode_data.squeeze(),polyline_feature],1)
        return output

class Subgraph(nn.Module):
    def __init__(self, input_channels):
        super(Subgraph, self).__init__()
        self.layer1 = subgraph_layer(input_channels)
        self.layer2 = subgraph_layer()
        self.layer3 = subgraph_layer()

    def forward(self, inputs):
        '''
        :param input: trajectory input : (last_observe,6)
               map input: (18 , 8)
        :return output: (128,)
        '''
        out1 = self.layer1(inputs)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        kernel_size = out3.size()[0]
        maxpool = nn.MaxPool1d(kernel_size)
        polyline_feature = maxpool(out3.unsqueeze(1).transpose(0,2)).squeeze()
        return polyline_feature