import argparse
from pickletools import uint8
from unittest.mock import patch
import torch
from torchvision import transforms, datasets
import shutil
import sys
import os
import warnings
import torch.optim as optim
from torch.utils.data import Dataset

sys.path.append("../../src/pixelcnn")
sys.path.append("../../src/vqvae2")
sys.path.append("../../src/utils")
sys.path.append("./src")
sys.path.append("./")


from vqvae2_config import VQVAE2Config
from vqvae2_scheduler import CycleScheduler
from ProgressBar import ProgressBar
from imagenet_loader import get_imagenet_dataloader
import numpy as np

# from julia import Main as JL

# JL.include(os.path.join(os.path.dirname(__file__), "../../VariationalJuice.jl/src/VariationalJuice.jl"))
# JL.include(os.path.join(os.path.dirname(__file__), "./src/learn_patch_clt.jl"))



def get_dataloader(args, shuffle_train = True, shuffle_test = False, num_tr_samples = None, num_ts_samples = None):
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



def get_vqvae2_model(args):

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


    return model, device



def get_data_for_vqclusters(args,model,device,data_loader,num_samples,split):
    s = args.imagenet_size // args.patch_size
    yz_feats = torch.zeros([num_samples,s,s,args.embed_size])
    all_imgs = torch.zeros([num_samples,3,args.imagenet_size,args.imagenet_size]) 

    print("> sampling data ... <\n")

    with torch.no_grad():
        model.eval()
        s_idx = 0
        for imgs, _ in data_loader:
            e_idx = s_idx + imgs.size(0)
            imgs = imgs.float().to(device)
            all_imgs[s_idx:e_idx,:,:] = imgs.detach().cpu()
            quant = model.get_continous_features(imgs / 255.0)
            yz_feats[s_idx:e_idx,:,:] = quant.detach().permute(0,2,3,1).cpu()
            s_idx = e_idx

    sub_imgs = all_imgs.view(num_samples,3,s,args.patch_size,s,args.patch_size).permute(0,2,4,1,3,5).reshape(num_samples,s,s,3,args.patch_size,args.patch_size).reshape(-1,3,args.patch_size,args.patch_size)
    yz_feats = yz_feats.reshape(-1,args.embed_size)
    
    data_dir = f"../progressive_growing/data/data_imagenet{args.imagenet_size}/"
    np.save(data_dir + f"data_{split}.npy",sub_imgs.numpy().astype(np.uint8))
    if args.independent_decoder:
        np.save(data_dir + f"idfeat_{split}.npy",yz_feats.numpy().astype(np.float32))
    else:
        np.save(data_dir + f"convfeat_{split}.npy",yz_feats.numpy().astype(np.float32))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type = int, default = 0)
    parser.add_argument("--imagenet-size", "-img", type = int, default = 32)
    parser.add_argument("--batch-size", type = int, default = 1024)
    parser.add_argument("--patch-size","-p", type = int, default = 4, help="4 for imagenet32/cifar, 8 for imagenet64")
    parser.add_argument("--n-clusters","-c", type = int, default = 1024)
    parser.add_argument("--num-skipped-scales", type = int, default = 1)
    parser.add_argument("--independent-decoder","-id", action="store_true")
    parser.add_argument("--ll-ratio", "-r",type = float, default = 1e-6)
    parser.add_argument("--num-tr-samples", "-tr", type = int, default = 400000)
    parser.add_argument("--num-ts-samples", "-ts", type = int, default = 40000)
    parser.add_argument("--embed-size", type = int, default = 128)
    parser.add_argument("--data-path", default = "../data")




    args = parser.parse_args()

    # Task identifier
    args.log_path = f"../../train_logs/imgnet{args.imagenet_size}_logs/p{args.patch_size}_c{args.n_clusters}_r{args.ll_ratio}_id-{args.independent_decoder}_metric-ll"
    args.model_save_path = os.path.join(args.log_path, "ckpt.pt")


    # Select gpu
    # JL.select_gpu(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    
    # Get imagenet dataloaders
    train_loader, test_loader = get_dataloader(args, num_tr_samples = args.num_tr_samples, num_ts_samples = args.num_ts_samples, shuffle_test = True)


    # Initialize model & load ckpt
    assert os.path.exists(args.model_save_path), f"Model checkpoint file {args.model_save_path} not found"
    model, device = get_vqvae2_model(args)
    ckpt = torch.load(args.model_save_path, map_location = torch.device('cpu'))
    model.load_state_dict(ckpt["model"])


    #Get progressive growing dataset
    get_data_for_vqclusters(args,model,device,train_loader,args.num_tr_samples,split='trn')
    get_data_for_vqclusters(args,model,device,test_loader,args.num_ts_samples,split='val')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()