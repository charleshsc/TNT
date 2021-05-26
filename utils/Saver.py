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
            log_file.write(key + ':' + getattr(args, key) + '\n')
        log_file.close()