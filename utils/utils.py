import torch
import torch.nn.functional as F
import numpy as np


def find_closest_to_gt_location(candidate_targets, gt_location):
    '''
    :param candidate_targets: (bs, N, 2) torch.tensor
    :param gt_location:  (bs, 2 ) torch.tensor
    :return: u (bs, 2, ) torch.tensor, delta_xy (bs, 2) torch.tensor
    '''
    assert len(candidate_targets.size()) == 3 and len(gt_location.size()) == 2
    batch_size = candidate_targets.size()[0]
    N = candidate_targets.size()[1]
    tmp_gt_location = gt_location.repeat(1, N).reshape(candidate_targets.size()) # (bs, N ,2)
    x = candidate_targets - tmp_gt_location # (bs, N ,2)
    x = torch.square(x)
    x = torch.sum(x, dim=2) # (bs, N)
    min_id = torch.argmin(x,dim=-1) # (bs, )
    tmp_candidate_targets = candidate_targets.gather(dim=1,index=min_id.repeat(2,1).T.unsqueeze(1)).squeeze() # (bs, 2)
    delta_xy = gt_location - tmp_candidate_targets
    return tmp_candidate_targets, delta_xy, min_id

def select_top_K_trajectories(trajectory, score, K, min_distance):
    '''
    :param trajectory: (M, T, 2) torch.tensor 
    :param score: (M, ) torch.tensor
    :param K: scalar
    :return: selected top K trajectory (K, T, 2) List[np.ndarray]
    '''
    assert trajectory.size()[0] == score.size()[0]
    sort_indices = torch.sort(-score).indices
    res_trajectory = []
    res_trajectory.append(trajectory[sort_indices[0]].cpu().numpy())
    num = 1
    for i in range(1, sort_indices.size()[0]):
        tra = trajectory[sort_indices[i]].cpu().numpy()
        dis_to_selected = [distance_of_trajectory(tra, selected) for selected in res_trajectory]
        if np.min(dis_to_selected) > min_distance:
            res_trajectory.append(tra)
            num = num + 1

        if num == K:
            break
    return res_trajectory



def distance_of_trajectory(trajectory_1, trajectory_2):
    '''
    :param trajectory_1: (T, 2) ndarray
    :param trajectory_2: (T, 2) ndarray
    :return: distance scalar
    '''
    tra_1 = torch.from_numpy(trajectory_1)
    tra_2 = torch.from_numpy(trajectory_2)
    assert tra_1.size() == tra_2.size()
    distance = torch.sum(torch.square(tra_1-tra_2),dim=-1)
    return torch.max(distance).cpu().detach().numpy()

def copy_state_dict(cur_state_dict, pre_state_dict, prefix=''):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue