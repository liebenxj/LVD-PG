import numpy as np
import faiss
import os


def train_kmeans_model(train_features, n_clusters, gpu_id = 0, centroids = None):
    train_features = np.ascontiguousarray(train_features)
    kmeans = faiss.Clustering(train_features.shape[1], n_clusters)
    if centroids is not None:
        faiss.copy_array_to_vector(
            np.ascontiguousarray(centroids.astype(np.float32).reshape(-1)), kmeans.centroids)
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

    return labels + 1


def save_kmeans_model(centroids, model_path):
    file_name = os.path.join(model_path, "centroids.npz")
    np.savez(file_name, centroids = centroids)