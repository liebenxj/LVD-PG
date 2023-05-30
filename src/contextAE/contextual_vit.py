import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os
import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), "../maskedAE/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../pixelcnn/"))

from vit_models import MAEEncoder
from model import *
from utils import *
from nearest_embed import NearestEmbed
from contextual_pixelcnn import ContextualPixelCNN


class ContextualViT(nn.Module):
    def __init__(self, enc_config, dec_config):
        super(ContextualViT, self).__init__()

        self.enc_config = enc_config
        self.dec_config = dec_config
        self.patch_hw = self.enc_config.image_size // self.enc_config.patch_size
        
        self.vit_encoder = MAEEncoder(self.enc_config)

        self.feature_aggregators = []
        self.discrete_embed_layers = []
        self.pixelcnn_decoders = []
        for level in range(self.dec_config.num_levels):
            aggr_layer = nn.Conv2d(
                self.enc_config.hidden_size, 
                self.dec_config.context_size,
                kernel_size = 2**level,
                stride = 2**level
            )
            self.add_module(f"aggr_level_{level}", aggr_layer)
            self.feature_aggregators.append(aggr_layer)

            if dec_config.discretize_features:
                nearest_embed_layer = NearestEmbed(
                    num_embeddings = self.dec_config.n_clusters,
                    embeddings_dim = self.dec_config.context_size
                )
                self.add_module(f"discrete_embed_level_{level}", nearest_embed_layer)
                self.discrete_embed_layers.append(nearest_embed_layer)

            img_patch_size = self.enc_config.patch_size * 2**level
            n_downscale = min(int(np.log2(img_patch_size)) - 1, 2)
            if n_downscale == 1:
                n_downscale = 0

            pixelcnn_decoder = ContextualPixelCNN(
                nr_resnet = self.dec_config.num_res_layers,
                nr_filters = self.dec_config.hidden_size,
                nr_logistic_mix = self.dec_config.num_logistic_mix,
                input_channels = self.dec_config.num_channels,
                context_channels = self.dec_config.position_encoding_dim + self.dec_config.context_size,
                n_levels = 3,
                n_downscale = n_downscale
            )

            self.add_module(f"pixelcnn_{level}", pixelcnn_decoder)
            self.pixelcnn_decoders.append(pixelcnn_decoder)

        patch_hw = enc_config.image_size // enc_config.patch_size
        num_patches = patch_hw**2
        self.patch_pos_encodings = nn.Parameter(torch.rand([num_patches, dec_config.position_encoding_dim]))

        self.feature_aggregators = nn.ModuleList(self.feature_aggregators)
        self.discrete_embed_layers = nn.ModuleList(self.discrete_embed_layers)
        self.pixelcnn_decoders = nn.ModuleList(self.pixelcnn_decoders)

    def forward(self, img):
        vit_out = self.vit_encoder(img)
        vit_embeddings = vit_out.last_hidden_state # (B, num_patches, embed_dim)
        vit_embeddings = vit_embeddings.transpose(1, 2).reshape(
            -1, self.enc_config.hidden_size, self.patch_hw, self.patch_hw
        )
        
        bpds = []
        for level in range(self.dec_config.num_levels):
            patched_context = self.patch_context(
                self.feature_aggregators[level](vit_embeddings)
            ).flatten(0, 1)

            # Discretize the embedding if required
            if self.dec_config.discretize_features:
                patched_context, _ = self.discrete_embed_layers[level](patched_context)
            
            img_patch_size = self.enc_config.patch_size * 2**level
            patched_img = self.patch_img(img, img_patch_size).flatten(0, 1)

            pos_embeddings = self.patch_pos_encodings[None, :, :].repeat(img.size(0), 1, 1).flatten(0, 1)

            with torch.no_grad():
                idxs = torch.arange(0, patched_img.size(0)).float().multinomial(num_samples = img.size(0), replacement = False).long()

            patched_img, patched_context = patched_img[idxs, :, :, :], patched_context[idxs, :]
            pos_embeddings = pos_embeddings[idxs, :]

            patched_context = torch.cat((patched_context, pos_embeddings), dim = 1)

            rec_features = self.pixelcnn_decoders[level](patched_img, patched_context)
            aveg_ll = discretized_mix_logistic_lls(patched_img, rec_features).mean()
            aveg_bpd = -aveg_ll / np.log(2.0) / patched_img.size(1) / patched_img.size(2) / patched_img.size(3)
            bpds.append(aveg_bpd)

        return torch.stack(bpds, dim = 0)

    def get_contextual_embeddings(self, img, level = None):
        vit_out = self.vit_encoder(img)
        vit_embeddings = vit_out.last_hidden_state # (B, num_patches, embed_dim)
        vit_embeddings = vit_embeddings.transpose(1, 2).reshape(
            -1, self.enc_config.hidden_size, self.patch_hw, self.patch_hw
        ) # (B, embed_dim, patch_h, patch_w)

        if level is not None:
            patched_context = self.patch_context(
                self.feature_aggregators[level](vit_embeddings)
            )
            return patched_context
        else:
            patched_contextes = dict()
            for level in range(self.dec_config.num_levels):
                patched_context = self.patch_context(
                    self.feature_aggregators[level](vit_embeddings)
                )
                img_patch_size = self.enc_config.patch_size * 2**level
                patched_contextes[img_patch_size] = patched_context
            return patched_contextes

    def patch_img(self, img, patch_size):
        B, C = img.size(0), img.size(1)
        patch_h, patch_w = img.size(2) // patch_size, img.size(3) // patch_size
        return img.reshape(B, C, patch_h, patch_size, patch_w, patch_size).permute(0, 1, 3, 5, 2, 4).reshape(
            B, C, patch_size, patch_size, patch_h * patch_w).permute(0, 4, 1, 2, 3)

    def patch_context(self, context):
        B, C = context.size(0), context.size(1)
        return context.reshape(B, C, -1).permute(0, 2, 1)


if __name__ == "__main__":
    from context_ae_config import MAE_ViT_encoder_config, PixelCNN_decoder_config

    device = torch.device("cuda:0")

    enc_config = MAE_ViT_encoder_config()
    dec_config = PixelCNN_decoder_config()
    model = ContextualViT(enc_config, dec_config)
    model.to(device)

    out = model(torch.zeros([32, 3, 32, 32]).to(device))
    print(out)