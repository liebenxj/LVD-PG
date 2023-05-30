import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets
import sys
import os
import io
import contextlib
from sklearn.cluster import KMeans as skKMeans
import faiss
import time
from copy import deepcopy

sys.path.append("../../src/maskedAE")
sys.path.append("../../src/utils")
sys.path.append("../../../src/maskedAE")
sys.path.append("../../../src/utils")
sys.path.append("../../../src/pixelcnn")
sys.path.append("../../../src/vqvae2")

from imagenet_loader import RandomSampler
from ProgressBar import ProgressBar
from sklearn.cluster import KMeans
import numpy as np
from sklearn.mixture import GaussianMixture
#imagenet64: from vqvae2_p8 import VQVAE2
# from vqvae2_p4 import VQVAE2
# from vqvae2_config import VQVAE2Config
import pickle



def get_patch_hw(model):
    if hasattr(model, "enc_config"):
        return model.enc_config.image_size // model.enc_config.patch_size
    elif hasattr(model, "config"):
        return model.config.image_size // model.config.patch_size
    else:
        raise NotImplementedError("Cannot find model configuration")


def get_img_features(model, device, data_loader, target_patch = (1, 1), visible_patches = [(1, 1)],
                     source_model = "MAE", return_imgs = False):
    if source_model == "MAE":
        return get_img_features_mae(model, device, data_loader, target_patch, visible_patches, use_pixel_features = False,
                                    return_imgs = return_imgs)
    elif source_model == "pixel":
        return get_img_features_mae(model, device, data_loader, target_patch, visible_patches, use_pixel_features = True,
                                    return_imgs = return_imgs)
    elif source_model == "VQVAE":
        return get_img_features_vqvae(model, device, data_loader, target_patch, return_imgs = return_imgs)
    else:
        raise NotImplementedError("Unknown source model {}".format(source_model))


def get_img_features_mae(model, device, data_loader, target_patch = (1, 1), visible_patches = [(1, 1)], 
                         use_pixel_features = False, return_imgs = False):
    
    sampler = data_loader.sampler
    num_samples = sampler.num_samples if isinstance(sampler, RandomSampler) else len(data_loader.dataset)
    
    # Convert to python format (count from 0)
    patch_hw = model.enc_config.image_size // model.enc_config.patch_size
    visible_patches = sorted([(patch[0]-1, patch[1]-1) for patch in visible_patches], key = lambda x: x[0] * patch_hw + x[1])
    target_patch = (target_patch[0]-1, target_patch[1]-1)

    patch_masks = torch.ones([patch_hw, patch_hw], dtype = torch.bool)
    for patch in visible_patches:
        patch_masks[patch[0], patch[1]] = False
    patch_masks = patch_masks.reshape(patch_hw * patch_hw).to(device)
    
    target_idx = visible_patches.index(target_patch)
    
    with torch.no_grad():
        progress = ProgressBar(total_epochs = 1, total_batches = int(np.ceil(num_samples / data_loader.batch_size)), statistics_name = [])
        progress.new_epoch_begin()

        if return_imgs:
            all_imgs = np.zeros([num_samples, 3, model.enc_config.image_size, model.enc_config.image_size], dtype = np.uint8)

        if not use_pixel_features:
            patch_features = torch.zeros([num_samples, model.enc_config.hidden_size])
        else:
            patch_features = torch.zeros([num_samples, 3 * model.enc_config.patch_size**2])
            img_x_s, img_x_e = target_patch[0] * model.enc_config.patch_size, (target_patch[0] + 1) * model.enc_config.patch_size
            img_y_s, img_y_e = target_patch[1] * model.enc_config.patch_size, (target_patch[1] + 1) * model.enc_config.patch_size
        s_idx = 0
        for imgs, _ in data_loader:
            e_idx = s_idx + imgs.size(0)
            if return_imgs:
                all_imgs[s_idx:e_idx,:,:,:] = imgs.detach().cpu().numpy().astype(np.uint8)
            imgs = imgs.float().to(device) / 255.0

            if not use_pixel_features:
                f = model.get_patch_features(imgs, patch_masks)
                patch_features[s_idx:e_idx,:] = f[:,target_idx,:]
            else:
                patch_features[s_idx:e_idx,:] = imgs[:,:,img_x_s:img_x_e,img_y_s:img_y_e].reshape(-1, 3 * model.enc_config.patch_size**2)

            s_idx = e_idx
            progress.new_batch_done()
        patch_features = patch_features.detach().cpu().numpy()
    
    progress.epoch_ends()
    
    if return_imgs:
        return patch_features, all_imgs
    else:
        return patch_features


def get_img_features_vqvae(model, device, data_loader, target_patch = (1, 1), return_imgs = False):
    # Convert to python format (count from 0)
    patch_hw = model.config.image_size // model.config.patch_size
    target_patch = (target_patch[0]-1, target_patch[1]-1) 
    


    sampler = data_loader.sampler
    num_samples = sampler.num_samples if isinstance(sampler, RandomSampler) else len(data_loader.dataset)


    with torch.no_grad():
        progress = ProgressBar(total_epochs = 1, total_batches = int(np.ceil(num_samples / data_loader.batch_size)), statistics_name = [])
        progress.new_epoch_begin()

        if return_imgs:
            all_imgs = np.zeros([num_samples, 3, model.config.image_size, model.config.image_size], dtype = np.uint8)


        f = torch.zeros([num_samples,patch_hw,patch_hw], dtype = torch.long)        
        patch_features = torch.zeros([num_samples], dtype = torch.long)

    
        s_idx = 0
        for imgs, _ in data_loader:
            e_idx = s_idx + imgs.size(0)
            if return_imgs:
                all_imgs[s_idx:e_idx,:,:,:] = imgs.detach().cpu().numpy().astype(np.uint8)
            imgs = imgs.float().to(device) / 255.0

            f[s_idx:e_idx] = model.get_patch_features(imgs)
            patch_features[s_idx:e_idx] = f[s_idx:e_idx, target_patch[0], target_patch[1]]

            s_idx = e_idx
            progress.new_batch_done()

        patch_features = patch_features.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
    progress.epoch_ends()



    if return_imgs:
        return patch_features, all_imgs
    else:
        return patch_features


def discretize_features(train_features, n_clusters, eval_features_set = [], method = "Kmeans", module = "faiss", gpu_id = 0):
    return get_kmeans_features(train_features, n_clusters, eval_features_set, module, gpu_id)


def get_kmeans_features(train_features, n_clusters, eval_features_set = [], module = "faiss", gpu_id = 0):
    assert n_clusters > 1, "`n_clusters` should be greater than 1"
    print("Running Kmeans... ", end = "")
    s = time.time()
    obj = None
    if module == "sklearn":
        kmeans = skKMeans(
            n_clusters = n_clusters
        )
        kmeans = kmeans.fit(train_features)
    
        results = []
        for features in eval_features_set:
            kmeans_features = kmeans.predict(features)
            results.append(kmeans_features)

    elif module == "faiss":
        train_features = np.ascontiguousarray(train_features)
        kmeans = faiss.Clustering(train_features.shape[1], n_clusters)
        kmeans.verbose = False
        kmeans.niter = 200
        kmeans.nredo = 5
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_id
        index = faiss.GpuIndexFlatL2(
            faiss.StandardGpuResources(),
            train_features.shape[1],
            cfg
        )
        kmeans.train(train_features, index)
        centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_clusters, train_features.shape[1])

        results = []
        for features in eval_features_set:
            features = np.ascontiguousarray(features.astype(np.float32))
            index = faiss.IndexFlatL2(train_features.shape[1])
            index.add(centroids)
            distances, labels = index.search(features, 1)
            labels = labels.ravel()

            results.append(labels)
    e = time.time()
    print("done ({:.2f}s)".format(e - s))
    if obj is not None:
        print(" > Obj: {:.4f}".format(obj))

    return tuple(results)





def remove_item(arr, item):
    idxs = np.where(arr == item)
    for idx in idxs:
        arr = np.delete(arr, idx)
    return arr




def train_kmeans_model(train_features, n_clusters, gpu_id = 0):
    train_features = np.ascontiguousarray(train_features)
    kmeans = faiss.Clustering(train_features.shape[1], n_clusters)
    kmeans.verbose = False
    kmeans.niter = 200
    kmeans.nredo = 5
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = gpu_id
    index = faiss.GpuIndexFlatL2(
        faiss.StandardGpuResources(),
        train_features.shape[1],
        cfg
    )
    kmeans.train(train_features, index)
    centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_clusters, train_features.shape[1])

    return centroids


def pred_kmeans_clusters(centroids, features):
    features = np.ascontiguousarray(features.astype(np.float32))
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(np.ascontiguousarray(centroids))
    _, labels = index.search(features, 1)
    labels = labels.ravel()

    return labels


def save_kmeans_model(centroids, model_path):
    file_name = os.path.join(model_path, "centroids.npz")
    np.savez(file_name, centroids = centroids)


def load_kmeans_model(model_path):
    file_name = os.path.join(model_path, "centroids.npz")
    centroids = np.load(file_name)["centroids"]

    return centroids





def train_patch_kmeans(model, device, train_loader, n_clusters, target_patch_idxs, visible_patches_dict, 
                       task_identifier, num_samples = None, source_model = "MAE"):
    if source_model == "MAE":
        return train_patch_kmeans_mae(model, device, train_loader, n_clusters, target_patch_idxs, visible_patches_dict,
                                      task_identifier, num_samples = num_samples, use_pixel_features = False)
    elif source_model == "pixel":
        return train_patch_kmeans_mae(model, device, train_loader, n_clusters, target_patch_idxs, visible_patches_dict,
                                      task_identifier, num_samples = num_samples, use_pixel_features = True)
    elif source_model == "VQVAE":
        return [None for _ in range(len(target_patch_idxs))]
    else:
        raise NotImplementedError("Unknown source model {}".format(source_model))


def train_patch_kmeans_mae(model, device, train_loader, n_clusters, target_patch_idxs, visible_patches_dict, 
                           task_identifier, num_samples = None, use_pixel_features = False):
    if num_samples is not None:
        sampler = RandomSampler(
            dataset = train_loader.dataset,
            num_samples = num_samples
        )
        train_loader = DataLoader(
            dataset = train_loader.dataset,
            batch_size = train_loader.batch_size,
            shuffle = False,
            sampler = sampler
        )

    kmeans_models = []

    base_dir = os.path.join("./temp/patch_kmeans_models/", task_identifier)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    for idx, target_patch in enumerate(target_patch_idxs):
        curr_dir = os.path.join(base_dir, f"{idx}")
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)
        if len(os.listdir(curr_dir)) == 0:
            assert isinstance(target_patch, tuple)
            print(f"> Training Kmeans for patch ({target_patch[0]}, {target_patch[1]})")
            
            print("  - computing MAE features...")
            tr_patch_features = get_img_features(model, device, train_loader, target_patch = target_patch, 
                                                 visible_patches = visible_patches_dict[target_patch], 
                                                 source_model = "MAE" if not use_pixel_features else "pixel")
            print(f"  - # examples: {tr_patch_features.shape[0]}")
            
            centroids = train_kmeans_model(tr_patch_features, n_clusters = n_clusters)

            cluster_ids = pred_kmeans_clusters(centroids, tr_patch_features)
            cluster_count = np.zeros([n_clusters], dtype = np.int64)
            for cluster_id in range(n_clusters):
                cluster_count[cluster_id] = np.sum((cluster_ids == cluster_id).astype(np.int64))
            print(f"  - min/max samples per cluster: {np.min(cluster_count)}, {np.max(cluster_count)}")

            save_kmeans_model(centroids, curr_dir)
        else:
            centroids = load_kmeans_model(curr_dir)
            print(f"> Loaded Kmeans for patch ({target_patch[0]}, {target_patch[1]})")

        kmeans_models.append(centroids)

    return kmeans_models


class DummyFile(object):
    def write(self, x):
        pass


@contextlib.contextmanager
def nostdout():
    saved_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = saved_stdout


@contextlib.contextmanager
def noop():
    yield


def get_data_for_cluster(mae_model, cluster_model, device, data_loader, cluster_id, image_size,
                         minimum_num_samples, target_patch, visible_patches, gmm_threshold = 0.1, 
                         from_julia = False, model_type = "Kmeans"):
    data = np.zeros([minimum_num_samples, 3, image_size, image_size], dtype = np.uint8)
    weights = np.zeros([minimum_num_samples], dtype = np.float32)

    if from_julia:
        cluster_id -= 1

    progressbar = ProgressBar(total_epochs = 1, total_batches = minimum_num_samples, statistics_name = [])

    s_idx = 0
    count = 0
    progressbar.new_epoch_begin()
    while s_idx < minimum_num_samples:
        with nostdout():
            patch_features, imgs = get_img_features(
                mae_model, device, data_loader, target_patch = target_patch, 
                visible_patches = visible_patches, return_imgs = True
            )
            cluster_ids = pred_kmeans_clusters(centroids = cluster_model, features = patch_features)
            filter = (cluster_ids == cluster_id)

        num_target_samples = np.sum(filter.astype(np.int32))
        if num_target_samples == 0:
            continue

        e_idx = s_idx + num_target_samples
        if num_target_samples > minimum_num_samples:
            data = np.zeros([num_target_samples, 3, image_size, image_size], dtype = np.uint8)
            weights = np.zeros([num_target_samples], dtype = np.float32)
            s_idx = 0
            e_idx = num_target_samples

        elif e_idx > minimum_num_samples:
            s_idx = minimum_num_samples - num_target_samples
            e_idx = minimum_num_samples

        progressbar.new_batch_done(n = num_target_samples)

        data[s_idx:e_idx,:,:,:] = imgs[filter,:,:,:]
        if model_type == "GMM":
            weights[s_idx:e_idx] = cluster_probs[filter, cluster_id]
        elif model_type == "Kmeans":
            weights[s_idx:e_idx] = 1.0
        else:
            raise NotImplemented(f"Unknown model type `{model_type}`")

        s_idx = e_idx
        count += 1

        if count > 100:
            print("Failing to collect enough clustered data")

    progressbar.epoch_ends()

    return data, weights


def subsample_data_loader(data_loader, num_samples, shuffle = False, get_data = False):
    for imgs, _ in data_loader:
        img_size = imgs.size()[1:]
    data = torch.zeros([num_samples, img_size[0], img_size[1], img_size[2]], dtype = torch.uint8)

    s_idx = 0
    while s_idx < num_samples:
        for imgs, _ in data_loader:
            e_idx = s_idx + imgs.size(0)
            if e_idx > num_samples:
                s_idx = num_samples - imgs.size(0)
                e_idx = num_samples

            data[s_idx:e_idx,:,:,:] = imgs.detach().cpu()

            s_idx = e_idx

            if s_idx >= num_samples:
                break

    if get_data:
        return data.detach().cpu().numpy()

    dataset = TensorDataset(data, torch.zeros([num_samples]))
    sampled_data_loader = DataLoader(
        dataset = dataset,
        batch_size = data_loader.batch_size,
        shuffle = shuffle
    )

    return sampled_data_loader


def get_data_for_all_clusters(mae_model, cluster_model, device, data_loader, n_clusters, image_size,
                              minimum_num_samples, target_patch, visible_patches, gmm_threshold = 0.1, 
                              from_julia = False, model_type = "Kmeans", max_ncounts = 100, source_model = "MAE"):
    if source_model == "MAE":
        return get_data_for_all_clusters_mae(
            mae_model, cluster_model, device, data_loader, n_clusters, image_size,
            minimum_num_samples, target_patch, visible_patches, gmm_threshold = gmm_threshold, 
            from_julia = from_julia, model_type = model_type, max_ncounts = max_ncounts, use_pixel_features = False
        )
    elif source_model == "pixel":
        return get_data_for_all_clusters_mae(
            mae_model, cluster_model, device, data_loader, n_clusters, image_size,
            minimum_num_samples, target_patch, visible_patches, gmm_threshold = gmm_threshold, 
            from_julia = from_julia, model_type = model_type, max_ncounts = max_ncounts, use_pixel_features = True
        )
    elif source_model == "VQVAE":
        return get_data_for_all_clusters_vqvae(
            mae_model, device, data_loader, n_clusters, image_size,
            minimum_num_samples, target_patch, visible_patches, max_ncounts = 100
        )
    else:
        raise NotImplementedError("Unknown source model {}".format(source_model))


def get_data_for_all_clusters_mae(mae_model, cluster_model, device, data_loader, n_clusters, image_size,
                                  minimum_num_samples, target_patch, visible_patches, gmm_threshold = 0.1, 
                                  from_julia = False, model_type = "Kmeans", max_ncounts = 100, use_pixel_features = False):
    data = np.zeros([n_clusters, minimum_num_samples, 3, image_size, image_size], dtype = np.uint8)
    weights = np.zeros([n_clusters, minimum_num_samples], dtype = np.float32)

    progressbar = ProgressBar(total_epochs = 1, total_batches = n_clusters * minimum_num_samples, statistics_name = [])

    s_idxs = np.zeros([n_clusters], dtype = np.int32)
    count = 0
    progressbar.new_epoch_begin()
    last_progress = 0
    while np.any(s_idxs < minimum_num_samples):
        with nostdout():
            patch_features, imgs = get_img_features(
                mae_model, device, data_loader, target_patch = target_patch, 
                visible_patches = visible_patches, return_imgs = True, use_pixel_features = use_pixel_features
            )

            if model_type == "Kmeans":
                cluster_ids = pred_kmeans_clusters(centroids = cluster_model, features = patch_features)
            else:
                raise NotImplemented(f"Unknown model type `{model_type}`")

        for cluster_id in range(n_clusters):

            s_idx = s_idxs[cluster_id]
            if s_idx >= minimum_num_samples:
                continue
            ####DEBUG####
            filter = (cluster_ids == cluster_id)

            num_target_samples = np.sum(filter.astype(np.int32))
            if num_target_samples == 0:
                continue

            e_idx = s_idx + num_target_samples
            if num_target_samples > minimum_num_samples:
                s_idx = 0
                e_idx = minimum_num_samples
                filter = np.where(filter == True)[0][:minimum_num_samples]
                num_target_samples = minimum_num_samples
            elif e_idx > minimum_num_samples:
                s_idx = minimum_num_samples - num_target_samples
                e_idx = minimum_num_samples

            data[cluster_id,s_idx:e_idx,:,:,:] = imgs[filter,:,:,:]
            if model_type == "GMM":
                weights[cluster_id,s_idx:e_idx] = cluster_probs[filter, cluster_id]
            elif model_type == "Kmeans":
                weights[cluster_id,s_idx:e_idx] = 1.0
            else:
                raise NotImplemented(f"Unknown model type `{model_type}`")

            s_idxs[cluster_id] = e_idx

        curr_progress = np.sum(s_idxs)
        cum_nsamples = curr_progress - last_progress
        progressbar.new_batch_done(n = cum_nsamples)
        last_progress = curr_progress

        count += 1

        if count > max_ncounts:
            break

    progressbar.epoch_ends()

    return data, weights, s_idxs


def get_data_for_all_clusters_vqvae(mae_model, device, data_loader, n_clusters, image_size,
                                    minimum_num_samples, target_patch, visible_patches, max_ncounts = 100):
    data = np.zeros([n_clusters, minimum_num_samples, 3, image_size, image_size], dtype = np.uint8)
    weights = np.zeros([n_clusters, minimum_num_samples], dtype = np.float32)

    progressbar = ProgressBar(total_epochs = 1, total_batches = n_clusters * minimum_num_samples, statistics_name = [])

    s_idxs = np.zeros([n_clusters], dtype = np.int32)
    count = 0
    progressbar.new_epoch_begin()
    last_progress = 0
    with nostdout():
        cluster_ids, imgs = get_img_features(
            mae_model, device, data_loader, target_patch = target_patch, 
            visible_patches = visible_patches, return_imgs = True, source_model = "VQVAE"
        )

    while np.any(s_idxs < minimum_num_samples):
        for cluster_id in range(n_clusters):
            s_idx = s_idxs[cluster_id]
            if s_idx >= minimum_num_samples:
                continue

            filter = (cluster_ids == cluster_id)

            num_target_samples = np.sum(filter.astype(np.int32))
            if num_target_samples == 0:
                continue

            e_idx = s_idx + num_target_samples
            if num_target_samples > minimum_num_samples:
                s_idx = 0
                e_idx = minimum_num_samples
                filter = np.where(filter == True)[0][:minimum_num_samples]
                num_target_samples = minimum_num_samples
            elif e_idx > minimum_num_samples:
                s_idx = minimum_num_samples - num_target_samples
                e_idx = minimum_num_samples

            data[cluster_id,s_idx:e_idx,:,:,:] = imgs[filter,:,:,:]
            weights[cluster_id,s_idx:e_idx] = 1.0

            s_idxs[cluster_id] = e_idx

        curr_progress = np.sum(s_idxs)
        cum_nsamples = curr_progress - last_progress
        if cum_nsamples == 0:
            break

        progressbar.new_batch_done(n = cum_nsamples)
        last_progress = curr_progress
        count += 1

        if count > max_ncounts:
            break

    progressbar.epoch_ends()

    return data, weights, s_idxs


def get_cluster_ids(mae_model, cluster_model, device, data_loader, target_patch, visible_patches, model_type = "Kmeans", suppress_print = False,
                    source_model = "MAE"):
    context = nostdout if suppress_print else noop
    with context():
        patch_features = get_img_features(
            mae_model, device, data_loader, target_patch = target_patch, 
            visible_patches = visible_patches, source_model = source_model
        )

        if source_model == "VQVAE":
            cluster_ids = patch_features
        elif model_type == "Kmeans":
            cluster_ids = pred_kmeans_clusters(centroids = cluster_model, features = patch_features)
        else:
            raise NotImplementedError()

    return cluster_ids



