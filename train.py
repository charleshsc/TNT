import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.ArgoverseDataset import ArgoverseForecastDataset
from modeling.TNT import TNT
from utils.args import obtain_env_args
from utils.Saver import Saver
from utils.eval_forecasting import compute_forecasting_metrics
from utils.utils import copy_state_dict
import numpy as np
import random

from tqdm import tqdm
import sys
import os
import logging
from torch.utils.tensorboard import SummaryWriter

# Manual Seed
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = obtain_env_args()
    writer = SummaryWriter(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    #### preparation ####
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    setup_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    #### dataset ####
    argo_dst = ArgoverseForecastDataset(args.last_observe, args.train_data_locate)
    train_loader = DataLoader(dataset=argo_dst, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    val_argo_dst = ArgoverseForecastDataset(args.last_observe, args.val_data_locate)
    val_loader = DataLoader(dataset=val_argo_dst, batch_size=args.batch_size, num_workers=args.num_worker)

    #### model ####
    model = TNT(traj_features=args.traj_features,map_features=args.map_features,args=args)
    model.to(torch.device(args.device))

    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.978)

    #### saver ####
    saver = Saver(args.save,args)

    best_pred = {'minADE': [-1, 999], 'minFDE': [-1, 999], 'MR': [-1, 999]}
    start_epochs = 0

    #### resume ####
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epochs = checkpoint['epoch']
        copy_state_dict(model.state_dict(), checkpoint['model_state_dict'])
        if not args.ft:
            copy_state_dict(optimizer.state_dict(),checkpoint['optimizer'])

        best_pred = checkpoint['pred_best']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))


    logging.info("Start Training")
    for epochs in range(start_epochs, args.epochs):
        logging.info("New epochs: "+str(epochs))
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        ## training
        train(model, args, argo_dst, train_loader, optimizer, writer, epochs)
        torch.cuda.empty_cache()
        lr_policy.step()

        ## validation
        with torch.no_grad():
            metric_results = infer(model, args, val_argo_dst, val_loader, epochs)

        logging.info('Training Epoch %d/%d: minADE: %.4f, minFDE: %.4f, MR: %.4f' % (epochs, args.epochs, metric_results["minADE"], metric_results["minFDE"], metric_results["MR"]))

        cur_save_model = False
        if metric_results["minADE"] < best_pred["minADE"][1]:
            best_pred["minADE"] = [epochs+1, metric_results["minADE"]]
            cur_save_model = True
        if metric_results["minFDE"] < best_pred["minFDE"][1]:
            best_pred["minFDE"] = [epochs+1, metric_results["minFDE"]]
            cur_save_model = True
        if metric_results["MR"] < best_pred["MR"][1]:
            best_pred["MR"] = [epochs+1, metric_results["MR"]]
            cur_save_model = True

        #### save the model ####
        if (epochs + 1) % args.epochs_to_save == 0 or cur_save_model:
            filename = "model_epoch_" + str(epochs + 1)
            state = {
                'epoch': epochs + 1,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pred_best': best_pred
            }
            saver.save_checkpoint(state, filename)

    logging.info("Finish Training")

def train(model, args, argo_dst, train_loader, optimizer, writer, epochs):
    #### one epoch training ####
    device = torch.device(args.device)
    print_every = args.steps_to_print

    # bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(train_loader,  ncols=80)
    for i, (traj_batch, map_batch) in enumerate(pbar):
        model.train()
        traj_batch = traj_batch.to(device=device, dtype=torch.float)
        candidate_targets = [argo_dst.generate_centerlines_uniform(city, args.N).to(device=device, dtype=torch.float)
                             for city in map_batch['city_name']]
        assert len(candidate_targets) == traj_batch.size()[0]
        batch_size = traj_batch.size()[0]

        optimizer.zero_grad()

        loss = model(traj_batch, map_batch, candidate_targets)
        pbar.set_description(
            "[Training Epoch %d/%d: step %d/%d, loss: %.4f]" % (epochs + 1, args.epochs, i + 1, len(train_loader), loss.item()))

        parameters = [p for p in model.parameters() if p.grad is not None]
        for p in parameters:
            p.grad.detach().div_(batch_size)

        optimizer.step()

        if (i + 1) % print_every == 0:
            logging.info('Training Epoch %d/%d: Iteration %d, loss = %.4f' % (epochs + 1, args.epochs, i + 1, loss.item()))
            writer.add_scalar("training_loss", loss.item(), epochs + 1)

        del traj_batch, map_batch, loss, candidate_targets
        torch.cuda.empty_cache()

    pbar.close()
    torch.cuda.empty_cache()


def infer(model, args, argo_dst, val_loader, epochs):
    #### one epoch val ####
    device = torch.device(args.device)

    # bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(val_loader,  ncols=80)
    final_res, final_gt, final_city_name = dict(),dict(),dict()
    for i, (traj_batch, map_batch) in enumerate(pbar):
        model.eval()
        traj_batch = traj_batch.to(device=device, dtype=torch.float)
        candidate_targets = [argo_dst.generate_centerlines_uniform(city, args.N).to(device=device, dtype=torch.float)
                             for city in map_batch['city_name']]
        assert len(candidate_targets) == traj_batch.size()[0]

        res, gt, city_name= model(traj_batch, map_batch, candidate_targets)
        final_res.update(res)
        final_gt.update(gt)
        final_city_name.update(city_name)

        pbar.set_description(
            "[Val Epoch %d/%d: step %d/%d]" % (
            epochs + 1, args.epochs, i + 1, len(val_loader)))

    pbar.close()


    metric_results = compute_forecasting_metrics(final_res, final_gt, final_city_name, model.K, model.T, model.min_distance)
    return metric_results

if __name__ == '__main__':
    main()