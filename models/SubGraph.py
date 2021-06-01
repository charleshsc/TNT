import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import numpy as np

class subgraph_layer(nn.Module):
    def __init__(self, input_channels=128, hidden_channels=64):
        super(subgraph_layer, self).__init__()
        self.fc = nn.Linear(input_channels, hidden_channels) # single fully connected layer
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, inputs):
        '''
        :param input: trajectory input : (bs, last_observe,6 | 128)
                      map input: (1, 18 , 8 | 128)
        :return output: (bs|1, last_observe | 18 , 128)
        '''
        assert len(inputs.size()) == 3
        hidden = self.fc(inputs)
        encode_data = F.relu(F.layer_norm(hidden, hidden.size()[1:])) # (bs|1, last_obeserve | 18, 64)
        kernel_size = encode_data.size()[1]  # last_observe | 18
        maxpool = nn.MaxPool1d(kernel_size)
        polyline_feature = maxpool(encode_data.transpose(1,2)).squeeze(dim=2)
        polyline_feature = polyline_feature.repeat(1, kernel_size).reshape(encode_data.size()) # (bs|1, last_obeserve | 18, 64)
        output = torch.cat([encode_data,polyline_feature],2) # (bs|1, last_obeserve | 18, 128)
        return output

class Subgraph(nn.Module):
    def __init__(self, input_channels):
        super(Subgraph, self).__init__()
        self.layer1 = subgraph_layer(input_channels)
        self.layer2 = subgraph_layer()
        self.layer3 = subgraph_layer()

    def forward(self, inputs):
        '''
        :param input: trajectory input : (bs,last_observe,6)
               map input: (1, 18 , 8)
        :return output: (bs|1, 128,)
        '''
        out1 = self.layer1(inputs)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2) # (bs|1, last_observe | 18 , 128)
        kernel_size = out3.size()[1]
        maxpool = nn.MaxPool1d(kernel_size)
        polyline_feature = maxpool(out3.transpose(1,2)).squeeze(2)
        return polyline_feature