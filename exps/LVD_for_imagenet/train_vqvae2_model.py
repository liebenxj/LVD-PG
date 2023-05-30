from tabnanny import check
from sklearn.utils import shuffle
import torch
from torchvision import transforms, datasets
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from torch import nn
import itertools



sys.path.append("../../src/vqvae2")
sys.path.append("../../src/pixelcnn")
sys.path.append("../../src/utils")
sys.path.append("../../src")
sys.path.append("../")
sys.path.append("./")




from vqvae2_scheduler import CycleScheduler
from vqvae2_config import VQVAE2Config
from ProgressBar import ProgressBar
from imagenet_loader import get_imagenet_dataloader
from torch.utils.data import TensorDataset, DataLoader, Dataset





def get_dataloader(args, shuffle_train = True, shuffle_test = False, num_tr_samples = None, num_ts_samples = None, test_only = False):
    if not test_only:
        train_loader = get_imagenet_dataloader(
            img_size = args.imagenet_size,
            train = True,
            batch_size = args.batch_size,
            shuffle = shuffle_train and num_tr_samples is None,
            prefix_path = args.data_path,
            num_samples = num_tr_samples
        )

    test_loader = get_imagenet_dataloader(
        img_size = args.imagenet_size,
        train = False,
        batch_size = args.batch_size,
        shuffle = shuffle_test and num_ts_samples is None,
        prefix_path = args.data_path,
        num_samples = num_ts_samples
    )

    return train_loader, test_loader


def get_model(args, train_loader):


    if args.patch_size == 4:
            from vqvae2_p4 import VQVAE2
    elif args.patch_size == 8:
        from vqvae2_p8 import VQVAE2
        
    vq_config = VQVAE2Config(image_size = args.imagenet_size, patch_size = args.patch_size, 
                            n_clusters = args.n_clusters, num_skipped_scales = args.num_skipped_scales)
    vq_config.independent_decoder = args.independent_decoder
    vq_config.ll_ratio = args.ll_ratio

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")

    model = VQVAE2(vq_config)
    model.to(device)

    if not hasattr(args, "num_epochs"):
        args.num_epochs = 1
    
    optimizer = optim.Adam(model.parameters(), lr = vq_config.lr)
    scheduler = CycleScheduler(
        optimizer, vq_config.lr, n_iter = len(train_loader) * args.num_epochs, momentum = None, warmup_proportion = 0.05
    )

    return model, optimizer, scheduler, device


def get_criterions_mse(output, imgs):
    with torch.no_grad():
        mae = torch.abs(output["x_recon"] - imgs).mean()
        maxabserr = torch.abs(output["x_recon"] - imgs).flatten(1).max(dim = 1)[0].mean()

    return [
        mae.detach().cpu().numpy().item(),
        maxabserr.detach().cpu().numpy().item(),
        output["latent_loss"].detach().cpu().numpy().item()
    ]

def get_criterions_ll(output,image_size):
    bpd = -output["lls"].detach().cpu().numpy().mean() / np.log(2.0) / (image_size*image_size*3)

    return [
        output["loss"].detach().cpu().numpy().item(),
        bpd,
        output["latent_loss"].detach().cpu().numpy().item()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type = int, default = 0)
    parser.add_argument("--imagenet-size", "-img", type = int, default = 32)
    parser.add_argument("--batch-size","-b", type = int, default = 64)
    parser.add_argument("--num-epochs", type = int, default = 50)
    parser.add_argument("--patch-size","-p", type = int, default = 4, help="4 for imagenet32/cifar, 8 for imagenet64")
    parser.add_argument("--n-clusters", "-c", type = int, default = 1024)
    parser.add_argument("--num-skipped-scales", type = int, default = 0)
    parser.add_argument("--mode", type = str, default = "train")
    parser.add_argument("--independent-decoder","-id", action="store_true")
    parser.add_argument("--load-model", default = True)
    parser.add_argument("--ll-ratio", type = float, default = 1e-6)
    parser.add_argument("--lr", type = float, default = 3e-4)
    parser.add_argument("--metric", type = str, default = "ll")
    parser.add_argument("--data-path", default = "../data")





    args = parser.parse_args()
    args.log_path = f"../../train_logs/imgnet{args.imagenet_size}_logs/p{args.patch_size}_c{args.n_clusters}_r{args.ll_ratio}_id-{args.independent_decoder}_metric-{args.metric}"
    args.model_save_path = os.path.join(args.log_path, "ckpt.pt")
    args.res_save_path = os.path.join(args.log_path, "res.txt")

    
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)


    train_loader, test_loader = get_dataloader(args)
    model, optimizer, scheduler, device = get_model(args, train_loader)

    # Load checkpoint
    epoch_start = 0
    if os.path.exists(args.model_save_path) and args.load_model:
        checkpoint = torch.load(args.model_save_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        epoch_start = checkpoint["epoch"]
        print("> Model loaded")

    if args.mode == "train":
        if args.metric == "mse":
            progressbar = ProgressBar(args.num_epochs, len(train_loader), statistics_name = ["mae", "maxabserr","vq_loss"], cumulate_statistics = True)
        elif args.metric == "ll":
            progressbar = ProgressBar(args.num_epochs, len(train_loader), statistics_name = ["ll_loss", "bpd", "latent_loss"], 
                                    cumulate_statistics = True)
        progressbar.set_epoch_id(epoch_start)
        best_test_mae = 1.0
        best_bpd = 10.0
        for epoch in range(epoch_start, args.num_epochs + 1):
            progressbar.new_epoch_begin()
            for imgs, _ in train_loader:
                imgs = imgs.float().to(device) / 255.0
                imgs_ll = 2 * (imgs-0.5)

                optimizer.zero_grad()

                
                if args.metric == "mse":
                    output = model(imgs)
                    crits = get_criterions_mse(output, imgs)

                elif args.metric == "ll":
                    output = model(imgs, imgs_ll)
                    crits = get_criterions_ll(output,args.imagenet_size)

                output["loss"].backward()

                optimizer.step()
                scheduler.step()

                progressbar.new_batch_done(crits)
            progressbar.epoch_ends()

            total_mae = 0.0
            total_maxabserr = 0.0
            total_ll_loss = 0.0
            total_bpd = 0.0
            total_vq_loss = 0.0
            for imgs, _ in test_loader:
                with torch.no_grad():
                    imgs = imgs.float().to(device) / 255.0
                    imgs_ll = 2 * (imgs-0.5)
                    if args.metric == "mse":
                        output = model(imgs)
                        crits = get_criterions_mse(output, imgs)
                        total_mae += crits[0]
                        total_maxabserr += crits[1]
                        total_vq_loss += crits[2]

                    elif args.metric == "ll":
                        output = model(imgs, imgs_ll)
                        crits = get_criterions_ll(output,args.imagenet_size)
                        total_ll_loss += crits[0]
                        total_bpd += crits[1]
                        total_vq_loss += crits[2]
                    
                    
            test_mae = total_mae / len(test_loader)
            test_maxabserr = total_maxabserr / len(test_loader)
            test_ll_loss = total_ll_loss / len(test_loader)
            test_bpd = total_bpd / len(test_loader)
            test_vq_loss = total_vq_loss / len(test_loader)
            if args.metric == "mse":
                res_str = "[Test] Epoch {:4d} - mae: {:.4f} - maxabserr: {:.4f} - vqloss: {:.4f}\n".format(epoch, test_mae, test_maxabserr, test_vq_loss)
            elif args.metric == "ll":
                res_str = "[Test] Epoch {:4d} - ll_loss: {:.4f} - bpd: {:.4f}/{:.4f} - vqloss: {:.4f}\n".format(epoch, test_ll_loss, test_bpd, best_bpd, test_vq_loss)
            print(res_str)
            with open(args.res_save_path,'a') as f:
                f.write(res_str)

            if (args.metric == "mse" and test_mae < best_test_mae) or (args.metric == "ll" and test_bpd < best_bpd):
                best_test_mae = test_mae
                best_bpd = test_bpd
                checkpoint = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict()
                }
                torch.save(checkpoint, args.model_save_path)

            if test_bpd < best_bpd:
                best_bpd = test_bpd
                checkpoint = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict()
                }
                torch.save(checkpoint, args.model_save_path)

    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))


if __name__ == "__main__":
    main()