import torch
from torch.utils.data import DataLoader
from Dataset.ArgoverseDataset import ArgoverseForecastDataset
from models.TNT import TNT
from utils.args import obtain_env_args
from eval.eval_forecasting import compute_forecasting_metrics
from utils.utils import copy_state_dict

from tqdm import tqdm
import sys
import os

city_lanenum = {
            'PIT' : 4891,
            'MIA' : 12417
            }

def main():
    args = obtain_env_args()

    #### preparation ####
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    val_argo_dst = ArgoverseForecastDataset(args.last_observe, args.test_data_locate)
    val_loader = DataLoader(dataset=val_argo_dst, batch_size=args.batch_size, num_workers=args.num_worker,pin_memory=True)

    model = TNT(traj_features=args.traj_features, map_features=args.map_features, args=args)
    model.to(torch.device(args.device))

    if args.resume is None:
        raise ValueError("please give the resume model ")
    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)

    copy_state_dict(model.state_dict(), checkpoint['model_state_dict'])

    print("=> loaded checkpoint '{}' ".format(args.resume))

    model.eval()
    metric_results = infer(model, args, val_argo_dst,val_loader)
    for key, val in metric_results.items():
        print(key + ": "+str(val))

def infer(model, args, argo_dst, val_loader):
    #### one epoch val ####
    device = torch.device(args.deivce)

    pbar = tqdm(val_loader, ncols=80)
    final_res, final_gt, final_city_name = dict(),dict(),dict()
    for i, (traj_batch, map_batch, city_names) in enumerate(pbar):
        model.eval()
        traj_batch = traj_batch.to(device=device, dtype=torch.float)
        vectormap_batch = []
        for batch, name in zip(map_batch, city_names):
            vectormap_batch.append(batch[:city_lanenum[name]].to(device=device, dtype=torch.float))
        candidate_targets = [argo_dst.generate_centerlines_uniform(city, args.N).to(device=device, dtype=torch.float)
                             for city in city_names]
        assert len(candidate_targets) == traj_batch.size()[0]

        res, gt, city_name= model(traj_batch, vectormap_batch, city_names, candidate_targets)
        final_res.update(res)
        final_gt.update(gt)
        final_city_name.update(city_name)

    pbar.close()

    T = args.total_step - args.last_observe
    metric_results = compute_forecasting_metrics(final_res, final_gt, final_city_name, args.K, T, args.miss_threshold)
    return metric_results

if __name__ == '__main__':
    main()