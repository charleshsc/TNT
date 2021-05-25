from Motion_Estimation import Motion_Estimator
from Target_Prediction import Target_predictor
from SubGraph import Subgraph
from GlobalGraph import GraphAttentionNet
from Trajectory_Scoring import Trajectory_Scorer
from utils import args

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TNT(nn.Module):
    def __init__(self,traj_features=6, map_features=8, args = args.args):
        super(TNT, self).__init__()
        #### hyper parameter from args ####
        self.M = args.M
        self.N = args.N
        self.alpha = args.alpha
        self.last_observe = args.last_observe
        self.total_step = args.total_step
        self.T = self.total_step - self.total_step  # for the predictor

        #### vector net component ####
        self.trajectory_subgraph = Subgraph(traj_features)
        self.map_subgraph = Subgraph(map_features)
        self.global_graph = GraphAttentionNet()

        #### Target prediction component ####
        self.target_predictor = Target_predictor(M=self.M, N=self.N)

        #### Motion Estimation component ####
        self.motion_estimator = Motion_Estimator(output_channels=2*self.T)

        #### Trajectory scoring component ####
        self.trajectory_scoere = Trajectory_Scorer(input_channels=2*self.T + 64, M=self.M, alpha=self.alpha)

    def forward(self, trajectory_batch, vectormap_batch):
        if self.training:
            self.forward_train(trajectory_batch, vectormap_batch)
        else:
            self.forward_val(trajectory_batch, vectormap_batch)


    def forward_train(self, trajectory_batch, vectormap_batch):
        # TODO:
        return None

    def forward_val(self, trajectory_batch, vectormap_batch):
        # TODO:
        return None

    def _loss(self, trajectory_batch, vectormap_batch):
        # TODO:
        return None


