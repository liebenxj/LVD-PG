import argparse
import torch
import torch.utils.data
from torch import dropout, nn, optim
import torch.nn.functional as F
from torchvision import *
import numpy as np
from scipy.special import softmax, logsumexp
import math
import time
import os
import itertools
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

from ProgressBar import ProgressBar

from julia import Main as JL

JL.include(os.path.join(os.path.dirname(__file__), "../../VariationalJuice.jl/src/VariationalJuice.jl"))


class PCVAE_categorical(nn.Module):
    def __init__(self, x_shape, pcs, z_shape = None, encoder = None, decoder = None, color_depth = 8):
        super(PCVAE_categorical, self).__init__()

        self.x_shape = x_shape
        self.z_shape = z_shape if z_shape is not None else self.x_shape
        self.num_x_vars = np.prod(self.x_shape)
        self.num_z_vars = np.prod(self.z_shape)
        self.color_depth = color_depth

        pc_z, pc_xz, pcxz2pcz = pcs
        mbpc_z, mbpc_xz, mbpc_mapping = JL.to_gpu(pc_z, pc_xz, pcxz2pcz)
        self.mbpc_z = mbpc_z
        self.mbpc_xz = mbpc_xz
        self.mbpc_mapping = mbpc_mapping
        self.pc_nparams = JL.num_parameters(self.mbpc_xz)
        self.pc_max_ncats = JL.num_categories(self.mbpc_z)

        # Prior parameters
        norm_prior_params = JL.vectorize_parameters(self.mbpc_z).reshape(1, -1)
        self.prior_params = JL.normalize_params(self.mbpc_z, norm_prior_params, undo = True)

        # PC decoder parameters
        self.normalized_leaf_params = torch.exp(torch.from_numpy(JL.leaf_params(self.mbpc_xz, self.num_x_vars, self.pc_max_ncats).reshape(
            self.x_shape[1], self.x_shape[2], self.pc_max_ncats, 2**self.color_depth)))

        self.encoder = encoder
        self.decoder = decoder

        self.device = torch.device("cpu")
        self.encoder_optimizer = None
        self.decoder_optimizer = None

    def to(self, device):
        super(PCVAE, self).to(device)

        self.device = device
        self.normalized_leaf_params = self.normalized_leaf_params.to(device)

    def init_optimizer(self, lr = 1e-3):
        if self.encoder is not None:
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr = lr)
        if self.decoder is not None:
            self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr = lr)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        b = z.size(0)
        y_logits1, switch = self.decoder(z)

        z = z.reshape(-1, self.x_shape[1], self.x_shape[2], self.pc_max_ncats)
        y_probs2 = torch.einsum("bhwi,hwij->bhwj", [z, self.normalized_leaf_params]) # (b, h, w, 2**color_depth)

        switch = switch.reshape(b, self.x_shape[1], self.x_shape[2], 1) * 0.9 + 0.1
        y_logits = y_logits1 * switch + torch.log(y_probs2 + 1e-8) * (1.0 - switch)

        return y_logits

    def get_normalized_flows(self, img):
        b = img.size(0)
        img_np = img.reshape(b, -1).detach().cpu().numpy()
        norm_flows = JL.per_sample_normalized_flows(self.mbpc_xz, img_np, batch_size = b)
        mapped_norm_flows = JL.map_pc_parameters(self.mbpc_z, self.mbpc_xz, norm_flows, param_mapping = self.mbpc_mapping)
        mapped_norm_flows = np.clip(mapped_norm_flows, -20.0, 0.0)
        mapped_norm_flows = JL.normalize_params(self.mbpc_z, mapped_norm_flows)

        return mapped_norm_flows

    def pc_sample(self, mapped_norm_flows, temperature = 1.0, no_gumbel = True):
        cat_params, _ = JL.gumbel_sample(
            self.mbpc_z, mapped_norm_flows, self.num_x_vars, self.pc_max_ncats, 
            temperature = temperature, norm_params = True, no_gumbel = no_gumbel
        )

        return cat_params

    def pc_kld(self, mapped_norm_flows):
        prior_params = np.repeat(self.prior_params, mapped_norm_flows.shape[0], axis = 0)
        klds, _ = JL.kld(self.mbpc_z, mapped_norm_flows, prior_params)

        return klds

    def pretrain_decoder(self, train_loader, test_loader, num_epochs, save_model = False):
        print("> Start pretraining decoder...")

        print("> Pre-computing normalized flows for training set...", end = "")

        train_cat_params = torch.zeros([len(train_loader.dataset), self.num_x_vars, self.pc_max_ncats], dtype = torch.float32)
        train_klds = torch.zeros([len(train_loader.dataset)], dtype = torch.float32)
        for item in train_loader:
            idxs, imgs = item[0], item[1]

            mapped_norm_flows = self.get_normalized_flows(imgs)
            cat_params = self.pc_sample(mapped_norm_flows)
            klds = self.pc_kld(mapped_norm_flows)

            train_cat_params[idxs, :, :] = torch.from_numpy(cat_params)
            train_klds[idxs] = torch.from_numpy(klds)

        print("done")

        print("> Pre-computing normalized flows for test set...", end = "")

        test_cat_params = torch.zeros([len(test_loader.dataset), self.num_x_vars, self.pc_max_ncats], dtype = torch.float32)
        test_klds = torch.zeros([len(test_loader.dataset)], dtype = torch.float32)
        for item in test_loader:
            idxs, imgs = item[0], item[1]

            mapped_norm_flows = self.get_normalized_flows(imgs)
            cat_params = self.pc_sample(mapped_norm_flows)
            klds = self.pc_kld(mapped_norm_flows)

            test_cat_params[idxs, :, :] = torch.from_numpy(cat_params)
            test_klds[idxs] = torch.from_numpy(klds)

        print("done")

        progress_bar = ProgressBar(num_epochs, len(train_loader), ["logp", "kld"])
        for epoch in range(1, num_epochs + 1):
            total_logp = 0.0
            total_kld = 0.0
            progress_bar.new_epoch_begin()
            for step_idx, item in enumerate(train_loader):
                idxs, imgs = item[0], item[1].to(self.device)
                cat_params = train_cat_params[idxs, :, :].to(self.device)
                b = imgs.size(0)
                
                # decode
                y_logits = self.decode(cat_params)
                y_probs = F.softmax(y_logits.reshape(b, self.x_shape[1], self.x_shape[2], 2**self.color_depth), dim = -1)
                data_onehot = F.one_hot(imgs.long(), num_classes = 256).reshape(b, self.x_shape[1], self.x_shape[2], 2**self.color_depth)
                loss = -torch.log((y_probs * data_onehot).sum(dim = 3) + 1e-6).sum(dim = (1,2)).mean()
                
                self.decoder_optimizer.zero_grad()
                loss.backward()
                self.decoder_optimizer.step()

                # kld
                klds = train_klds[idxs]
                aveg_kld = float(torch.mean(klds).detach().cpu().numpy().item())
                total_kld += aveg_kld

                logp = -float(loss.detach().cpu().numpy().item())
                total_logp += logp

                progress_bar.new_batch_done([total_logp / (step_idx + 1), total_kld / (step_idx + 1)])

            progress_bar.epoch_ends([total_logp / len(train_loader), total_kld / len(train_loader)])

            total_logp = 0.0
            total_kld = 0.0
            for step_idx, item in enumerate(test_loader):
                with torch.no_grad():
                    idxs, imgs = item[0], item[1].to(self.device)
                    cat_params = test_cat_params[idxs, :, :].to(self.device)
                    b = imgs.size(0)
                    
                    # decode
                    y_logits = self.decode(cat_params)
                    y_probs = F.softmax(y_logits.reshape(b, self.x_shape[1], self.x_shape[2], 2**self.color_depth), dim = -1)
                    data_onehot = F.one_hot(imgs.long(), num_classes = 256).reshape(b, self.x_shape[1], self.x_shape[2], 2**self.color_depth)
                    loss = -torch.log((y_probs * data_onehot).sum(dim = 3) + 1e-6).sum(dim = (1,2)).mean()

                    # kld
                    klds = test_klds[idxs]
                    aveg_kld = float(torch.mean(klds).detach().cpu().numpy().item())
                    total_kld += aveg_kld

                    logp = -float(loss.detach().cpu().numpy().item())
                    total_logp += logp

            print("> [Pretrain] Test logp: {:.4f} - kld: {:.4f} - elbo: {:.4f}".format(
                total_logp / len(test_loader), total_kld / len(test_loader), (total_logp - total_kld) / len(test_loader)))

            if save_model and epoch % 10 == 0:
                self.save()
