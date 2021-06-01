import os
import shutil
import torch
from collections import OrderedDict
import glob
import torch.distributed as dist

class Saver(object):

    def __init__(self, save_direcotry, args):
        self.experiment_dir = save_direcotry
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.args = args
        self.save_experiment_config(args)
        self.save_scripts()

    def save_checkpoint(self, state, filename='checkpoint.pth'):
        """Saves checkpoint to disk"""
        if filename.split('.')[-1] != 'pth':
            filename = filename + '.pth'
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        best_pred = state['pred_best']
        with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
            f.write(str(best_pred))



    def save_experiment_config(self, args):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')

        for key in vars(args):
            log_file.write(key + ':' + str(getattr(args, key)) + '\n')
        log_file.close()

    def save_scripts(self):
        scripts_to_save = []

        abs_dir = os.path.realpath(".")  # 当前的绝对位置
        root_name = self.args.root_name
        root_dir = abs_dir[:abs_dir.index(root_name) + len(root_name)]
        for name in os.listdir(root_dir):
            if name[0] != '.' and os.path.isdir(os.path.join(root_dir,name)):
                scripts_to_save = scripts_to_save + glob.glob(os.path.join(root_dir, name,'*.py'))
        scripts_to_save = scripts_to_save + glob.glob(os.path.join(root_dir,'*.py'))

        if scripts_to_save is not None:
            os.mkdir(os.path.join(self.experiment_dir, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(self.experiment_dir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)