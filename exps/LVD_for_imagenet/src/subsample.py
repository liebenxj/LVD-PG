import torch
from torch.utils.data import TensorDataset, DataLoader


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
