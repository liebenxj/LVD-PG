import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../pixelcnn/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../densenet/"))

from contextual_pixelcnn import ContextualPixelCNN
from densenet import DenseNet
from utils import *
from nearest_embed import NearestEmbed


class GumbelSoftmax(nn.Module):
    def __init__(self, dim = -1, hard = True):
        super(GumbelSoftmax, self).__init__()

        self.dim = dim
        self.hard = hard

    def forward(self, x):
        if self.training:
            return F.gumbel_softmax(x, dim = self.dim, hard = self.hard)
        else:
            assert self.dim == -1 or self.dim == x.dim() - 1
            return F.one_hot(torch.argmax(x, dim = self.dim), num_classes = x.size(self.dim)).float()


class SelfContextualViT(nn.Module):
    def __init__(self, config):
        super(SelfContextualViT, self).__init__()

        self.config = config

        n_downscale = min(int(np.log2(config.patch_size)) - 1, 2)
        if n_downscale == 1:
            n_downscale = 0

        self.context_encoder = DenseNet(
            image_size = config.patch_size,
            growth_rate = config.densenet_growth_rate,
            block_config = [config.densenet_block_size] * config.densenet_nblocks,
            num_init_features = config.densenet_ninit_features,
            compression = config.densenet_compression,
            num_output_features = config.context_dim
        )
        self.bn = nn.BatchNorm1d(config.context_dim)

        if config.discretize_features:
            self.nearest_embed = NearestEmbed(
                num_embeddings = config.n_clusters,
                embeddings_dim = config.context_dim
            )

        self.contextual_pixelcnn = ContextualPixelCNN(
            nr_resnet = config.num_res_layers,
            nr_filters = config.hidden_size, 
            nr_logistic_mix = config.num_logistic_mix, 
            input_channels = config.num_channels,
            context_channels = config.context_dim + config.position_encoding_dim, 
            n_downscale = n_downscale, 
            n_levels = 3
        )

        patch_hw = config.image_size // config.patch_size
        num_patches = patch_hw**2
        self.patch_pos_encodings = nn.Parameter(torch.rand([num_patches, config.position_encoding_dim]))
        self.patch_hw = patch_hw

    def forward(self, imgs):
        B = imgs.size(0)
        patched_imgs = self.patch_img(imgs).flatten(0, 2)
        pos_embeddings = self.patch_pos_encodings[None, :, :].repeat(B, 1, 1).flatten(0, 1)

        with torch.no_grad():
            idxs = torch.arange(0, patched_imgs.size(0)).float().multinomial(num_samples = B, replacement = False).long()

        patched_imgs, pos_embeddings = patched_imgs[idxs, :, :, :], pos_embeddings[idxs, :]

        context = self.context_encoder(patched_imgs)
        context = self.bn(context)
        if self.config.discretize_features:
            context, _ = self.nearest_embed(context)
        context = torch.cat((context, pos_embeddings), dim = 1)

        out = self.contextual_pixelcnn(patched_imgs, context)
        aveg_ll = discretized_mix_logistic_lls(patched_imgs, out).mean()
        aveg_bpd = -aveg_ll / np.log(2.0) / patched_imgs.size(1) / patched_imgs.size(2) / patched_imgs.size(3)
        return aveg_bpd

    def get_contextual_embeddings(self, imgs):
        B = imgs.size(0)
        patched_imgs = self.patch_img(imgs).flatten(0, 2)
        context = self.context_encoder(patched_imgs)
        if self.config.discretize_features:
            context = self.cont2desc(context)
        context = context.reshape(B, self.patch_hw, self.patch_hw, self.config.context_dim)

        return context

    def patch_img(self, imgs):
        B, C = imgs.size(0), imgs.size(1)
        patch_h, patch_w = imgs.size(2) // self.config.patch_size, imgs.size(3) // self.config.patch_size
        return imgs.reshape(B, C, patch_h, self.config.patch_size, patch_w, self.config.patch_size).permute(0, 2, 4, 1, 3, 5)
        


