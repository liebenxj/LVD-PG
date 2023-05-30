import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer
from dataclasses import dataclass
from typing import Optional, Tuple
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from vit_models import *


@dataclass
class ProbSequenceClassifierOutput():

    loss: Optional[torch.FloatTensor] = None
    dist_loss: Optional[torch.FloatTensor] = None
    ent_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ProbabilisticMaskedAE(nn.Module):
    def __init__(self, enc_config, dec_config, mask_generator, dist_temperature = 1.0):
        super(ProbabilisticMaskedAE, self).__init__()

        self.enc_config = enc_config
        self.dec_config = dec_config
        self.dist_temperature = dist_temperature

        self.patch_size = enc_config.patch_size
        self.image_size = enc_config.image_size
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.mask_generator = mask_generator

        self.vit = MAEEncoder(enc_config)

        self.encoder_to_decoder = nn.Linear(enc_config.hidden_size, dec_config.hidden_size, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_config.hidden_size))
        self.register_buffer(
            'decoder_position_embeddings', 
            self.get_sinusoid_encoding_table(self.num_patches + 1, dec_config.hidden_size)
        )
        self.decoder = MAEDecoder(dec_config)
        self.decoder_norm = nn.LayerNorm(dec_config.hidden_size, eps=dec_config.layer_norm_eps)
        self.decoder_head = nn.Linear(dec_config.hidden_size, self.patch_size ** 2 * dec_config.num_channels)

        self.mse_loss = nn.MSELoss()

        self.device = torch.device("cpu")

    def to(self, device):
        super(ProbabilisticMaskedAE, self).to(device)

        self.device = device

    def forward(self, imgs, patch_masks=None, head_mask=None, output_attentions=False,
                output_hidden_states=False, interpolate_pos_encoding=False, dist_batch_size=32):

        # Make masks if not provided explicitly
        if patch_masks is None:
            patch_masks = self.mask_generator()
            patch_masks = torch.from_numpy(patch_masks).to(self.device)

        # Encode
        outputs = self.vit(
            imgs,
            patch_masks=patch_masks,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding
        )

        sequence_output = outputs[0]
        sequence_output = self.encoder_to_decoder(sequence_output)
        batch_size, unmask_size, dim = sequence_output.shape

        normed_sequence_output = F.softmax(sequence_output, dim = 2)
        log_normed_sequence_output = F.log_softmax(sequence_output, dim = 2)

        # Compute input distance matrix
        with torch.no_grad():
            patches = self.img2patch(imgs) # (B, num_patches, n_pixels)
            if patch_masks.dim() == 1:
                patches = patches[:,patch_masks == 0,:]
            else:
                raise NotImplementedError()
            idxs = torch.arange(0, patches.size(0)).float().multinomial(num_samples = min(patches.size(0), dist_batch_size), replacement = False).long()
            patches = patches[idxs, :, :]

            patch_dists = torch.sqrt((patches.unsqueeze(0) - patches.unsqueeze(1)).pow(2).mean(dim = -1)) # (B, B, num_patches)
            vmin, vmax = torch.quantile(patch_dists, 0.1), torch.quantile(patch_dists, 0.9)
            patch_scores = torch.exp(-(patch_dists.clamp(vmin, vmax) - vmin) / (vmax - vmin) / self.dist_temperature)
        klds = (normed_sequence_output[idxs, :, :].unsqueeze(0) * (log_normed_sequence_output[idxs, :, :].unsqueeze(0) - \
                log_normed_sequence_output[idxs, :, :].unsqueeze(1))).sum(dim = -1) # (B, B, num_patches)
        dist_loss = (patch_scores * klds).mean() / patch_scores.mean()

        ent_loss = -(normed_sequence_output * log_normed_sequence_output).sum(dim = -1).mean()

        expand_pos_embed = self.decoder_position_embeddings[:, 1:].expand(batch_size, -1, -1).clone().detach()
        if len(patch_masks.size()) == 1:
            pos_emd_unmask = expand_pos_embed[:,~patch_masks,:].reshape(batch_size, -1, dim)
            pos_emd_mask = expand_pos_embed[:,patch_masks,:].reshape(batch_size, -1, dim)
        elif len(patch_masks.size()) == 2:
            assert torch.all(patch_masks.sum(dim = 1) == patch_masks[0,:].sum())
            pos_emd_unmask = expand_pos_embed[~patch_masks,:].reshape(batch_size, -1, dim)
            pos_emd_mask = expand_pos_embed[patch_masks,:].reshape(batch_size, -1, dim)
        else:
            raise ValueError(f"Does not support `patch_masks` with size {patch_masks.size()}")

        decoder_inputs = torch.cat([normed_sequence_output + pos_emd_unmask, self.mask_token + pos_emd_mask], dim=1)

        # Decode
        decoder_outputs = self.decoder(
            decoder_inputs,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        logits = self.decoder_head(self.decoder_norm(decoder_outputs[0]))
        recon_img = self.patch2img(logits)

        # Compute loss
        loss = self.mse_loss(recon_img, imgs)

        return ProbSequenceClassifierOutput(
            loss=loss,
            dist_loss=dist_loss,
            ent_loss=ent_loss,
            logits=recon_img,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_patch_features(self, imgs, patch_masks):
        outputs = self.vit(imgs, patch_masks = patch_masks)
        sequence_output = outputs[0] # (B, num_unmasked_patches, feature_dim)
        return sequence_output

    def img2patch(self, imgs, p = None):
        """
        imgs: (N, 3, H, W)
        """
        p = self.enc_config.patch_size if p is None else p
        c = self.enc_config.num_channels
        h = imgs.size(2) // p
        w = imgs.size(3) // p

        imgs = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = imgs.permute(0, 2, 4, 3, 5, 1).reshape(imgs.shape[0], h * w, p ** 2 * c)
        return x

    def patch2img(self, x, p = None):
        """
        x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
        """
        p = self.enc_config.patch_size if p is None else p
        c = self.enc_config.num_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    @staticmethod
    def get_sinusoid_encoding_table(n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)