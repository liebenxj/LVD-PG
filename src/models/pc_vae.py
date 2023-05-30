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


class PCVAE(nn.Module):
    def __init__(self, x_shape, pcs, z_shape = None, encoder = None, decoder = None, color_depth = 8):
        super(PCVAE, self).__init__()

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

        assert JL.num_variables(self.mbpc_z) == np.prod(self.z_shape), "{} neq {}".format(JL.num_variables(self.mbpc_z), np.prod(self.z_shape))

        self.pc_decoder_enabled = False
        self.mbpc_x_z = None
        self.marked_pc_idxs = None

        # Prior parameters
        norm_prior_params = JL.vectorize_parameters(self.mbpc_z).reshape(1, -1)
        self.prior_params = JL.normalize_params(self.mbpc_z, norm_prior_params, undo = True)

        self.encoder = encoder
        self.decoder = decoder

        self.device = torch.device("cpu")
        self.encoder_optimizer = None
        self.decoder_optimizer = None

    def to(self, device):
        super(PCVAE, self).to(device)

        self.device = device

    def init_optimizer(self, lr = 1e-3):
        if self.encoder is not None:
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr = lr)
        if self.decoder is not None:
            self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr = lr)

    def register_p_x_z_pc(self, pc_x_z, marked_nodes):
        self.pc_decoder_enabled = True
        self.mbpc_x_z = JL.CuMetaBitsProbCircuit(pc_x_z)
        self.marked_pc_idxs = JL.mark_nodes(self.mbpc_x_z, marked_nodes)

    def get_normalized_flows(self, img):
        b = img.size(0)
        img_np = img.reshape(b, -1).detach().cpu().numpy()
        mapped_norm_flows = JL.normalized_flows_with_prealloc(self.mbpc_xz, self.mbpc_z, img_np)

        return mapped_norm_flows

    def pc_sample(self, mapped_norm_flows, temperature = 1.0, no_gumbel = True):
        cat_params = JL.gumbel_sample_with_prealloc(
            self.mbpc_z, mapped_norm_flows, temperature = temperature, norm_params = True, no_gumbel = no_gumbel
        )

        return cat_params

    def pc_kld(self, mapped_norm_flows):
        prior_params = np.repeat(self.prior_params, mapped_norm_flows.shape[0], axis = 0)
        klds = JL.kld_with_prealloc(self.mbpc_z, mapped_norm_flows, prior_params)

        return klds

    def pc_p_x_z(self, cat_params):
        x_probs = JL.cond_td_prob_with_prealloc(self.mbpc_x_z, cat_params)
        
        return x_probs

    def pretrain_decoder(self, train_loader, test_loader, num_epochs, normalized_flows_file_name = None, 
                         save_model = False, preload_batch_size = 0, log_file_name = None):
        print("> Start pretraining decoder...")

        if normalized_flows_file_name is not None and os.path.exists(normalized_flows_file_name):
            data = np.load(normalized_flows_file_name)
            train_cat_params = torch.from_numpy(data["train_cat_params"])
            train_klds = torch.from_numpy(data["train_klds"])
            test_cat_params = torch.from_numpy(data["test_cat_params"])
            test_klds = torch.from_numpy(data["test_klds"])
            print("> Cached encoder outputs are loaded...")

        else:
            print("> Pre-computing normalized flows for training set...")

            if preload_batch_size > 0:
                orig_train_loader = train_loader
                orig_test_loader = test_loader
                train_loader = torch.utils.data.DataLoader(
                    dataset = orig_train_loader.dataset, 
                    batch_size = preload_batch_size,
                    shuffle = False,
                    drop_last = False
                )
                test_loader = torch.utils.data.DataLoader(
                    dataset = orig_test_loader.dataset, 
                    batch_size = preload_batch_size,
                    shuffle = False,
                    drop_last = False
                )

            JL.gumbel_sample_preallocation(self.mbpc_z, num_examples = train_loader.batch_size, num_vars = self.num_z_vars, par_buffer_size = self.pc_max_ncats)
            JL.kld_preallocation(self.mbpc_z, num_examples = train_loader.batch_size)
            JL.normalized_flows_preallocation(self.mbpc_xz, self.mbpc_z, num_examples = train_loader.batch_size, num_vars = self.num_x_vars, 
                                              mbpc_mapping = self.mbpc_mapping)
            if self.pc_decoder_enabled:
                JL.cond_td_prob_preallocation(self.mbpc_x_z, self.marked_pc_idxs[0], self.marked_pc_idxs[1], 
                                              (self.num_z_vars, self.pc_max_ncats), num_examples = train_loader.batch_size)

            train_cat_params = torch.zeros([len(train_loader.dataset), self.num_z_vars, self.pc_max_ncats], dtype = torch.float32)
            train_klds = torch.zeros([len(train_loader.dataset)], dtype = torch.float32)
            progress_bar = ProgressBar(1, len(train_loader), ["logp", "kld"], cumulate_statistics = True)
            progress_bar.new_epoch_begin()
            logp = 0.0
            for item in train_loader:
                idxs, imgs, b = item[0], item[1], item[1].size(0)

                mapped_norm_flows = self.get_normalized_flows(imgs)
                cat_params = self.pc_sample(mapped_norm_flows)
                klds = self.pc_kld(mapped_norm_flows)
                kld = klds.mean().item()

                train_cat_params[idxs, :, :] = torch.from_numpy(cat_params[:b,:,:])
                train_klds[idxs] = torch.from_numpy(klds[:b])

                if self.pc_decoder_enabled:
                    x_probs = torch.from_numpy(self.pc_p_x_z(cat_params[:b,:,:])) # (b, c*h*w, 2**cdepth)
                    imgs = imgs.reshape(b, -1).unsqueeze(-1)
                    imgs = imgs[:, :x_probs.size(1), :]
                    logps = torch.log(x_probs.gather(2, imgs.long()).squeeze()).sum(dim = 1)
                    logp = logps.mean().detach().cpu().numpy().item()

                progress_bar.new_batch_done([logp, kld])

            progress_bar.epoch_ends()

            print("> Pre-computing normalized flows for test set...")

            test_cat_params = torch.zeros([len(test_loader.dataset), self.num_z_vars, self.pc_max_ncats], dtype = torch.float32)
            test_klds = torch.zeros([len(test_loader.dataset)], dtype = torch.float32)
            progress_bar = ProgressBar(1, len(test_loader), ["logp", "kld"], cumulate_statistics = True)
            progress_bar.new_epoch_begin()
            logp = 0.0
            for item in test_loader:
                idxs, imgs, b = item[0], item[1], item[1].size(0)

                mapped_norm_flows = self.get_normalized_flows(imgs)
                cat_params = self.pc_sample(mapped_norm_flows)
                klds = self.pc_kld(mapped_norm_flows)

                test_cat_params[idxs, :, :] = torch.from_numpy(cat_params[:b,:,:])
                test_klds[idxs] = torch.from_numpy(klds[:b])

                if self.pc_decoder_enabled:
                    x_probs = torch.from_numpy(self.pc_p_x_z(cat_params[:b,:,:])) # (b, c*h*w, 2**cdepth)
                    imgs = imgs.reshape(b, -1).unsqueeze(-1)
                    imgs = imgs[:, :x_probs.size(1), :]
                    logps = torch.log(x_probs.gather(2, imgs.long()).squeeze()).sum(dim = 1)
                    logp = logps.mean().detach().cpu().numpy().item()

                progress_bar.new_batch_done([logp, kld])

            progress_bar.epoch_ends()

            JL.gumbel_sample_free_mem()
            JL.kld_free_mem()
            JL.normalized_flows_free_mem()
            if self.pc_decoder_enabled:
                JL.cond_td_prob_free_mem()

            if normalized_flows_file_name is not None:
                np.savez(
                    normalized_flows_file_name, 
                    train_cat_params = train_cat_params.detach().cpu().numpy(),
                    train_klds = train_klds.detach().cpu().numpy(),
                    test_cat_params = test_cat_params.detach().cpu().numpy(),
                    test_klds = test_klds.detach().cpu().numpy()
                )

            if preload_batch_size > 0:
                train_loader = orig_train_loader
                test_loader = orig_test_loader

        progress_bar = ProgressBar(num_epochs, len(train_loader), ["logp", "kld", "bpd"])
        for epoch in range(1, num_epochs + 1):
            total_logp = 0.0
            total_kld = 0.0
            progress_bar.new_epoch_begin()
            for step_idx, item in enumerate(train_loader):
                idxs, imgs = item[0], item[1].to(self.device)
                cat_params = train_cat_params[idxs, :, :].to(self.device)
                b = imgs.size(0)

                pz = torch.distributions.Categorical(probs = cat_params)
                z = F.one_hot(pz.sample(), num_classes = cat_params.size(2)).float()
                
                # decode
                logp = self.decoder.get_logp(z, imgs)
                loss = -logp
                
                # backpropagation step + lr decay
                self.update_decoder(loss)

                # kld
                klds = train_klds[idxs]
                aveg_kld = float(torch.mean(klds).detach().cpu().numpy().item())
                total_kld += aveg_kld

                logp = -float(loss.detach().cpu().numpy().item())
                total_logp += logp

                progress_bar.new_batch_done([total_logp / (step_idx + 1), total_kld / (step_idx + 1),
                    -(total_logp - total_kld) / np.prod(self.x_shape) / np.log(2.0) / (step_idx + 1)])

            logp = total_logp / len(train_loader)
            kld = total_kld / len(train_loader)
            bpd = -(total_logp - total_kld) / np.prod(self.x_shape) / np.log(2.0) / len(train_loader)
            progress_bar.epoch_ends([logp, kld, bpd])

            if log_file_name is not None:
                with open(log_file_name, "a+") as f:
                    f.write("[train] epoch {} - logp: {:.2f}, kld: {:.2f}, bpd: {:.4f}\n".format(epoch, logp, kld, bpd))

            total_logp = 0.0
            total_kld = 0.0
            for step_idx, item in enumerate(test_loader):
                with torch.no_grad():
                    idxs, imgs = item[0], item[1].to(self.device)
                    cat_params = test_cat_params[idxs, :, :].to(self.device)
                    b = imgs.size(0)
                    
                    # decode
                    logp = self.decoder.get_logp(cat_params, imgs)

                    # kld
                    klds = test_klds[idxs]
                    aveg_kld = float(torch.mean(klds).detach().cpu().numpy().item())
                    total_kld += aveg_kld

                    logp = float(logp.detach().cpu().numpy().item())
                    total_logp += logp

            logp = total_logp / len(test_loader)
            kld = total_kld / len(test_loader)
            bpd = -(total_logp - total_kld) / np.prod(self.x_shape) / np.log(2.0) / len(test_loader)
            print("> [Pretrain] Test logp: {:.4f} - kld: {:.4f} - bpd: {:.4f}".format(logp, kld, bpd))
            if log_file_name is not None:
                with open(log_file_name, "a+") as f:
                    f.write("[test] epoch {} - logp: {:.2f}, kld: {:.2f}, bpd: {:.4f}\n".format(epoch, logp, kld, bpd))

            if save_model and epoch % 10 == 0:
                self.save()

    def update_decoder(self, loss):
        self.update_lr(self.decoder_optimizer)

        self.decoder_optimizer.zero_grad()

        loss.backward()

        self.decoder_optimizer.step()

    def update_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            lr = self._lr_step(lr)
            param_group['lr'] = lr

    # learning rate schedule
    def _lr_step(self, curr_lr, decay = 0.999995, min_lr = 1e-5):
        # only decay after certain point
        # and decay down until minimal value
        if curr_lr > min_lr:
            curr_lr *= decay
            return curr_lr
        return curr_lr

    def save(self, file_name = "model.pt"):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name = "model.pt"):
        if os.path.exists(file_name):
            state_dict = torch.load(file_name, map_location = self.device)
            self.load_state_dict(state_dict, strict = False)
            print("> Model loaded")
        else:
            print("> Checkpoint file doesn't exist")
