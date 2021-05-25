import argparse
import glob
import os
import os.path as osp

def obtain_env_args():
    abs_dir = osp.realpath(".")  # 当前的绝对位置
    this_dir = abs_dir.split(osp.sep)[-1]
    root_dir = abs_dir[:abs_dir.index(this_dir)]
    directory = osp.join(root_dir,'run')

    runs = sorted(glob.glob(osp.join(directory, 'experiment_*')))
    run_id = max([int(x.split('_')[-1]) for x in runs]) + 1 if runs else 0
    if run_id != 0 and len(os.listdir(osp.join(directory, 'experiment_{}'.format(str(run_id - 1))))) == 0:
        run_id = run_id - 1
    experiment_dir = osp.join(directory, 'experiment_{}'.format(str(run_id)))

    if not osp.exists(experiment_dir):
        os.makedirs(experiment_dir)

    save = experiment_dir

    parser = argparse.ArgumentParser(description="TNT")
    parser.add_argument('--N', type=int, default=1000, help="over-sample number of target candidates")
    parser.add_argument('--M', type=int, default=50, help="keep a small number for processing")
    parser.add_argument('--alpha',type=float, default=0.01, help="the temperature in scoring psi")
    parser.add_argument('--last_observe',type=int, default=30, help="the last observe in the trajectory for the agent")
    parser.add_argument('--total_step', type=int, default=49,help="the total step in the trajectory for the agent")
    parser.add_argument('--save', type=str, default=save)
    parser.add_argument('--tau',type=float,default=0.02)
    parser.add_argument('--capacity',type=int, default=10000)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--actor_lr',type=float,default=1e-3)
    parser.add_argument('--critic_lr',type=float,default=1e-3)
    parser.add_argument('--gamma',type=float,default=0.99)
    parser.add_argument('--checkpoint',type=str,default=None)

    args = parser.parse_args()
    return args

args = obtain_env_args()


