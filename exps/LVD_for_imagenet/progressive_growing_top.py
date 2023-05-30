from julia import Main as JL
import argparse
import os
import sys
import torch 

JL.include(os.path.join(os.path.dirname(__file__), "../../VariationalJuice.jl/src/VariationalJuice.jl"))
JL.include(os.path.join(os.path.dirname(__file__), "../../VariationalJuice.jl/src-jl/LatentPCs.jl"))
JL.include(os.path.join(os.path.dirname(__file__), "./src/train_PG_top_level_pcs.jl"))
JL.include(os.path.join(os.path.dirname(__file__), "./src/learn_patch_clt.jl"))




sys.path.append("../../src/utils")
sys.path.append("../../src/pixelcnn")
sys.path.append("../../src/vqvae2")
sys.path.append("./src")
sys.path.append("./")


from vqvae2_scheduler import CycleScheduler
from vqvae2_config import VQVAE2Config
from imagenet_loader import get_imagenet_dataloader



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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type = int, default = 0)
    parser.add_argument("--num-independent-clusters", "-idc",type = int, default = 400)
    parser.add_argument("--num-init-clusters", "-i", type = int, default = 2)
    parser.add_argument("--num-final-clusters", "-f", type = int, default = 4)
    parser.add_argument("--imagenet-size", "-img", type = int, default = 32)
    parser.add_argument("--fname-idx", "-idx", type = int, default = 4)
    parser.add_argument("--batch-size", type = int, default = 256)
    parser.add_argument("--data-path", default = "../data")
    parser.add_argument("--patch-size","-p", type = int, default = 4)#imagenet_64:patch_size=8
    parser.add_argument("--n-clusters","-c", type = int, default = 1024)
    parser.add_argument("--independent-decoder","-id", action="store_true")
    parser.add_argument("--ll-ratio", "-r",type = float, default = 1e-6)
    parser.add_argument("--num-skipped-scales", type = int, default = 0)
    parser.add_argument("--num-tr-samples", "-tr", type = int, default = 200000)
    parser.add_argument("--num-ts-samples", "-ts", type = int, default = 40000)

    args = parser.parse_args()

    JL.select_gpu(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    args.log_path = f"../../train_logs/imgnet{args.imagenet_size}_logs/p{args.patch_size}_c{args.n_clusters}_r{args.ll_ratio}_id-{args.independent_decoder}_metric-ll"
    args.model_save_path = os.path.join(args.log_path, "ckpt.pt")


    assert os.path.exists(args.model_save_path), f"Model checkpoint file {args.model_save_path} not found"
    model, device = get_vqvae2_model(args)
    ckpt = torch.load(args.model_save_path, map_location = torch.device('cpu'))
    model.load_state_dict(ckpt["model"])

    train_loader, test_loader = get_dataloader(args, shuffle_train = True, shuffle_test = True)


    top_level_hclt_params = {
        "num_tr_samples": args.num_tr_samples,
        "num_ts_samples": args.num_ts_samples,
        "num_warmup_samples": args.num_tr_samples,
        "batch_size": args.batch_size,
        "num_epochs1": 10,
        "num_epochs2": 10,
        "pseudocount": 0.1,
        "param_inertia1": 0.9,
        "param_inertia2": 0.99,
        "param_inertia3": 0.999
    }


    patch_idxs, clt_edges, ancestors_dict, descendents_dict = JL.learn_clt_for_top_pc(
        model, device, train_loader, args.patch_size, patch_hw = args.imagenet_size//args.patch_size, patch_n = args.imagenet_size//(args.patch_size*2), patch_x1 = 0, 
        patch_y1 = 0, n_clusters = args.n_clusters, base_dir = args.log_path, 
        source_model = "VQVAE", from_py = True)
    print("Patch indices:", patch_idxs)
    print("Number of CLT edges:", len(clt_edges))
    print("CLT edges:", clt_edges)


    JL.training_pg_top_level_pcs(args.imagenet_size, args.fname_idx, args.patch_size, patch_idxs, clt_edges, train_loader, test_loader, args.num_independent_clusters, args.num_init_clusters, args.num_final_clusters, top_level_hclt_params)


if __name__ == "__main__":
    main()