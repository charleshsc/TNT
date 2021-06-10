import argparse
import glob
import os
import os.path as osp

import torch.cuda


def obtain_env_args():
    abs_dir = osp.realpath(".")  # 当前的绝对位置
    root_name = 'TNT'
    root_dir = abs_dir[:abs_dir.index(root_name)+len(root_name)]
    directory = osp.join(root_dir,'run')

    runs = sorted(glob.glob(osp.join(directory, 'experiment_*')))
    run_id = max([int(x.split('_')[-1]) for x in runs]) + 1 if runs else 0
    if run_id != 0 and len(os.listdir(osp.join(directory, 'experiment_{}'.format(str(run_id - 1))))) == 0:
        run_id = run_id - 1
    experiment_dir = osp.join(directory, 'experiment_{}'.format(str(run_id)))

    if not osp.exists(experiment_dir):
        os.makedirs(experiment_dir)

    save = experiment_dir

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    parser = argparse.ArgumentParser(description="TNT")
    parser.add_argument('--N', type=int, default=1000, help="over-sample number of target candidates")
    parser.add_argument('--M', type=int, default=50, help="keep a small number for processing")
    parser.add_argument('--alpha',type=float, default=0.01, help="the temperature in scoring psi")
    parser.add_argument('--last_observe',type=int, default=30, help="the last observe in the trajectory for the agent")
    parser.add_argument('--total_step', type=int, default=49,help="the total step in the trajectory for the agent")
    parser.add_argument('--save', type=str, default=save)
    parser.add_argument('--batch_size',type=int,default=2, help="the batch size for the training stage")
    parser.add_argument('--device',type=str,default=device,help="the device to train")
    parser.add_argument('--gpu',type=int, default=0, help="if the device is cuda then which card to use")
    parser.add_argument('--lambda_1',type=float,default=0.1, help="the weight for the loss1")
    parser.add_argument('--lambda_2',type=float,default=1.0, help="the weight for the loss2")
    parser.add_argument('--lambda_3',type=float,default=0.1, help="the weight for the loss3")
    parser.add_argument('--K',type=int,default=6, help="the final number of candidate trajectory")
    parser.add_argument('--min_distance',type=float, default=0.5, help="the min distance in selection stage")
    parser.add_argument('--seed',type=int, default=12345, help="the seed to init the random")
    parser.add_argument('--learning_rate',type=float,default=0.001,help="the learning rate for the optimizer")
    parser.add_argument('--train_data_locate',type=str,default="../train/data",help="the train dataset root directory")
    parser.add_argument('--val_data_locate',type=str,default="../val/data",help="the val dataset root directory")
    parser.add_argument('--test_data_locate', type=str, default="../test/data", help="the test dataset root directory")
    parser.add_argument('--num_worker',type=int,default=0,help="the num worker for the data loader")
    parser.add_argument('--traj_features',type=int,default=6,help="the feature dim for the trajectory")
    parser.add_argument('--map_features',type=int,default=8,help="the feature dim for the map")
    parser.add_argument('--epochs',type=int,default=50,help="the num epochs for the training")
    parser.add_argument('--steps_to_print',type=int,default=10,help="the steps to print the loss")
    parser.add_argument('--epochs_to_save',type=int,default=1,help="the num epochs to save the model")
    parser.add_argument('--resume',type=str,default=None,help="the pretrained model to reload")
    parser.add_argument('--ft',type=bool,default=True,help="fine-tuning in optimizer")
    parser.add_argument('--root_name',type=str,default=root_name,help="save the scripts helping")
    parser.add_argument('--miss_threshold', type=float,default=2.0, help="The miss threshold in the eval stage for the MR")

    args = parser.parse_args()
    return args

args = obtain_env_args()


