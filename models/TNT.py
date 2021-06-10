from models.Motion_Estimation import Motion_Estimator
from models.Target_Prediction import Target_predictor
from models.SubGraph import Subgraph
from models.GlobalGraph import GraphAttentionNet
from models.Trajectory_Scoring import Trajectory_Scorer
from utils import args
from utils.utils import find_closest_to_gt_location, select_top_K_trajectories

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
        self.T = self.total_step - self.last_observe  # for the predictor
        self.device = torch.device(args.device)
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.K = args.K
        self.min_distance = args.min_distance

        #### vector net component ####
        self.trajectory_subgraph = Subgraph(traj_features)
        self.map_subgraph = Subgraph(map_features)
        self.global_graph = GraphAttentionNet()

        #### Target prediction component ####
        self.target_predictor = Target_predictor(M=self.M, N=self.N)

        #### Motion Estimation component ####
        self.motion_estimator = Motion_Estimator(output_channels=2*self.T)

        #### Trajectory scoring component ####
        self.trajectory_scorer = Trajectory_Scorer(input_channels=2*self.T + 64, M=self.M, alpha=self.alpha)

    def forward(self, trajectory_batch, vectormap_batch, city_names, candidate_targets):
        if self.training:
            return self.forward_train(trajectory_batch, vectormap_batch, city_names, candidate_targets)
        else:
            return self.forward_val(trajectory_batch, vectormap_batch, city_names, candidate_targets)


    def forward_train(self, trajectory_batch, vectormap_batch, city_names, candidate_targets):
        '''
        :param trajectory_batch: (batch_size, 49, 6)
        :param vectormap_batch: List[(12417, 18, 8)] len=batch_size
        :param city_names: List[] len=batch_size
        :param candidate_targets: List[torch.tensor(N,2)] len=batch_size
        :return:
        '''
        batch_size = trajectory_batch.size()[0]
        candidate_targets = torch.cat([candi_targets.unsqueeze(0) for candi_targets in candidate_targets],dim=0)

        label = trajectory_batch[:, self.last_observe:, 2:4] # (bs, T, 2)
        origin_point = trajectory_batch[:, self.last_observe-1, 2:4].reshape(batch_size,2) # (bs, 2)

        context_feature_list = []

        #### Scene context encoding ####
        trajectory_feature = self.trajectory_subgraph(trajectory_batch[:, :self.last_observe])  # (bs, 128)

        assert len(vectormap_batch) == len(city_names)
        for i, city in enumerate(city_names):
            single_vec_map = vectormap_batch[i] #(12417 | 4891, 18, 8)
            map_feature = self.map_subgraph(single_vec_map)  # ( 12417 | 4891, 128)
            complex_feature = torch.cat([trajectory_feature[[i]], map_feature], dim=0) # (12418 | 4892, 128)

            polyline_feature = F.normalize(complex_feature, p=2, dim=1)  # L2 normalize

            global_context_feature = self.global_graph(polyline_feature)[[0]]  # (1, 64)
            context_feature_list.append(global_context_feature)

        context_feature = torch.cat(context_feature_list,0) #(bs, 64)

        #### Target prediction loss ####
        assert candidate_targets.size() == (batch_size, self.N, 2)
        target_prediction_x = context_feature.repeat(1, self.N).reshape((batch_size, self.N, -1)) # (bs, N, 64)
        gt_location = label[:,-1] # (bs, 2)
        u, delta_xy, idx = find_closest_to_gt_location(candidate_targets, gt_location)
        u.requires_grad_(False)
        delta_xy.requires_grad_(False)
        loss1 = self.target_predictor._loss(candidate_targets, target_prediction_x, u, delta_xy, idx)

        #### Motion Estimation ####
        label.requires_grad_(False)
        origin_point.requires_grad_(False)
        loss2 = self.motion_estimator._loss(gt_location, context_feature, label, origin_point)

        #### Trajectory Scoring ####
        M_idx, _, M_delta_xy = self.target_predictor.forward_M(candidate_targets, target_prediction_x) # (bs, M), (bs,M,2)
        M_candidate_target = torch.cat([(candidate_targets[i][M_idx[i]] + M_delta_xy[i]).unsqueeze(0) for i in range(batch_size)],dim=0 ) # (bs, M ,2)
        M_x = torch.cat([target_prediction_x[id][M_idx[id]].unsqueeze(0) for id in range(batch_size)],dim=0) #(bs, M ,64)
        M_trajectory = self.motion_estimator(M_candidate_target, M_x)  # (bs, M, T, 2)
        loss3 = self.trajectory_scorer._loss(M_trajectory, M_x, label)

        loss = self.lambda_1 * loss1 + self.lambda_2 * loss2 + self.lambda_3 * loss3

        if np.isnan(loss.item()):
            print(trajectory_batch[:, 0, -1]) # the file name
            raise Exception("Loss ERROR!")
        return loss


    def forward_val(self, trajectory_batch, vectormap_batch, city_names, candidate_targets):
        '''
        :param trajectory_batch: (batch_size, 49, 6)
        :param vectormap_batch: List[(12417, 18, 8)] len=batch_size
        :param city_names: List[] len=batch_size
        :param candidate_targets: List[torch.tensor(N,2)] len=batch_size
        :return: result {key : K_trajectory(List[np.ndarray](K,T,2)}, gt {key : trajecoty(np.ndarray(T,2)}
        '''
        batch_size = trajectory_batch.size()[0]
        candidate_targets = torch.cat([candi_targets.unsqueeze(0) for candi_targets in candidate_targets],dim=0)

        label = trajectory_batch[:, self.last_observe:, 2:4]  # (bs, T, 2)

        result, gt, city_name = dict(), dict(), dict()

        context_feature_list = []

        trajectory_feature = self.trajectory_subgraph(trajectory_batch[:, :self.last_observe])  # (bs, 128)

        for i, city in enumerate(city_names):
            single_vec_map = vectormap_batch[i]  # (12417 | 4891, 18, 8)
            map_feature = self.map_subgraph(single_vec_map)  # ( 12417 | 4891, 128)
            complex_feature = torch.cat([trajectory_feature[[i]], map_feature], dim=0)  # (12418 | 4892, 128)

            polyline_feature = F.normalize(complex_feature, p=2, dim=1)  # L2 normalize

            global_context_feature = self.global_graph(polyline_feature)[[0]]  # (1, 64)
            context_feature_list.append(global_context_feature)

        context_feature = torch.cat(context_feature_list, 0)  # (bs, 64)

        #### top M targets ####
        target_prediction_x = context_feature.repeat(1, self.N).reshape((batch_size, self.N, -1))  # (bs, N, 64)

        M_idx, _, M_delta_xy = self.target_predictor.forward_M(candidate_targets, target_prediction_x)  # (bs,M),(bs,M,2)
        M_candidate_target = torch.cat([(candidate_targets[i][M_idx[i]] + M_delta_xy[i]).unsqueeze(0) for i in range(batch_size)], dim=0)  # (bs, M ,2)
        M_x = torch.cat([target_prediction_x[id][M_idx[id]].unsqueeze(0) for id in range(batch_size)], dim=0)  # (bs, M ,64)
        M_trajectory = self.motion_estimator(M_candidate_target, M_x)  # (bs, M, T, 2)

        M_score = self.trajectory_scorer(M_trajectory, M_x)  # (bs, M)

        for i in range(batch_size):
            K_trajectory = select_top_K_trajectories(M_trajectory[i], M_score[i], self.K, self.min_distance)  # (K,T,2)

            key = trajectory_batch[i, 0, -1].int().item()
            result.update({key: K_trajectory})
            gt.update({key: label[i].cpu().numpy()})
            city_name.update({key: city_names[i]})

        return result,gt,city_name

