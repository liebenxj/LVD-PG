import numpy as np
import torch
import os
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
from PIL import Image


class ToTensor():
    def __call__(self, X_i):
        return torch.from_numpy(np.array(X_i, copy=True)).permute(2, 0, 1)


class RandomSampler(Sampler):
    def __init__(self, dataset, num_samples=None):
        self.dataset = dataset
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.dataset)
        return self._num_samples

    def __iter__(self):
        n = len(self.dataset)
        return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


class ImageNet(Dataset):
    def __init__(self, img_size = 32, root_dir = "/scratch/anji/data/imagenet32", train = True, transform = None):
        self.train = train
        if self.train:
            root_dir = os.path.join(root_dir, "train/")
        else:
            root_dir = os.path.join(root_dir, "val/")

        self.transform = transform
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)
        self.datasets = []
        self.labels = []
        self.img2dataset = dict()
        self.img2idx = dict()
        sample_idx = 0
        for file in self.files:
            print("> Loading {}".format(file))
            fname = os.path.join(self.root_dir, file)
            data = np.load(fname)
            self.datasets.append(data["data"].reshape(-1, 3, img_size, img_size))
            self.labels.append(data["labels"])
            for i in range(self.datasets[-1].shape[0]):
                self.img2dataset[sample_idx] = len(self.datasets) - 1
                self.img2idx[sample_idx] = i
                sample_idx += 1
                
        self.length = sample_idx

    def __getitem__(self, index):
        img = self.datasets[self.img2dataset[index]][self.img2idx[index]]
        label = self.labels[self.img2dataset[index]][self.img2idx[index]]

        img = torch.from_numpy(img).type(torch.uint8)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.length


def get_imagenet_dataloader(img_size = 32, train = True, batch_size = 64, shuffle = True, 
                            transform = None, prefix_path = "../data", num_samples = None):
    root_dir = os.path.join(prefix_path, "imagenet{}".format(img_size))
    dataset = ImageNet(img_size = img_size, root_dir = root_dir, train = train, transform = transform)

    if num_samples is not None:
        sampler = RandomSampler(dataset, num_samples = num_samples)
    else:
        sampler = None

    shuffle = shuffle and sampler is None


    data_loader = torch.utils.data.DataLoader(
        dataset = dataset, batch_size = batch_size, shuffle = shuffle, sampler = sampler
    )

 

    return data_loader