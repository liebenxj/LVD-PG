from argparse import Namespace
from unittest.mock import patch
import numpy as np
import torch
import torch.nn as nn
from transformers import ViTMAEForPreTraining
from torchvision import transforms
from transformers.models.vit_mae.modeling_vit_mae import get_2d_sincos_pos_embed_from_grid
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from vit_models import RandomMaskingGenerator

class PretrainedViTMAE(nn.Module):
    def __init__(self, image_size, patch_size, num_channels = 3, min_mask_ratio = 0.1, max_mask_ratio = 0.9):
        super(PretrainedViTMAE, self).__init__()

        if patch_size != 16:
            image_size = image_size * 16 / patch_size
            patch_size = 16

        self.config = Namespace()
        self.config.image_size = image_size
        self.config.patch_size = patch_size
        self.config.num_patches = (image_size // patch_size) ** 2
        self.config.num_channels = num_channels
        self.config.min_mask_ratio = min_mask_ratio
        self.config.max_mask_ratio = max_mask_ratio

        self.model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        self.mask_generator = RandomMaskingGenerator(self.config)
        self.resize_fn = transforms.Resize(image_size)

        self.sub_modules = {
            "vit_embedding": self.model.vit.embeddings,
            "vit_encoder": self.model.vit.encoder,
            "vit_layernorm": self.model.vit.layernorm,
            "decoder": self.model.decoder
        }

        # Customized position embedding
        self.embedding_dim = self.sub_modules["vit_embedding"].position_embeddings.shape[-1]
        self.register_buffer(
            "pos_embeddings",
            self.get_trained_pos_embed(
                self.embedding_dim,
                int(self.config.num_patches**0.5),
                add_cls_token = True
            )
        )
        
        decoder_embed_dim = self.model.decoder.decoder_pos_embed.shape[-1]
        self.model.decoder.decoder_pos_embed = nn.Parameter(
                self.get_2d_sincos_pos_embed(
                decoder_embed_dim,
                int(self.config.num_patches**0.5),
                add_cls_token = True
            )
        )

        self.device = torch.device("cpu")

    def to(self, device):
        super(PretrainedViTMAE, self).to()

        self.device = device

    def forward(self, imgs, patch_masks = None):
        B, C, H, W = imgs.shape
        if C != self.config.num_channels:
            raise ValueError(f"Input images should have {self.config.num_channels} channels.")
        if H != self.config.image_size or W != self.config.image_size:
            imgs = self.resize_fn(imgs)

        # Make masks if not provided explicitly
        if patch_masks is None:
            patch_masks = self.mask_generator()
            patch_masks = torch.from_numpy(patch_masks).to(self.device)

        vit_embedding = self.sub_modules["vit_embedding"]
        vit_encoder = self.sub_modules["vit_encoder"]
        vit_layernorm = self.sub_modules["vit_layernorm"]
        decoder = self.sub_modules["decoder"]

        # Embed patches
        embeddings = vit_embedding.patch_embeddings.projection(imgs).flatten(2).transpose(1, 2)
        embeddings = embeddings + self.pos_embeddings[:, 1:, :]

        # Apply patch masks
        if patch_masks.dim() == 1:
            patch_masks = patch_masks.unsqueeze(0).repeat(B, 1)
        embeddings, _, ids_restore = self.random_masking(embeddings, noise = patch_masks)

        # Append cls token
        cls_token = vit_embedding.cls_token + vit_embedding.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim = 1)

        # Apply encoder
        encoder_outputs = vit_encoder(embeddings)
        sequence_output = encoder_outputs[0]
        sequence_output = vit_layernorm(sequence_output)

        # Apply decoder
        decoder_outputs = decoder(sequence_output, ids_restore)
        logits = decoder_outputs.logits

        recon_imgs = self.patch2img(logits)

        img_masks = 1.0 - self.patch2img(
            patch_masks.unsqueeze(-1).repeat(1, 1, logits.size(2)).float()
        )

        return recon_imgs, img_masks

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.
        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = seq_length - noise[0,:].sum()

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, mask, ids_restore

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, add_cls_token = False):
        orig_grid_size = (self.sub_modules["vit_embedding"].patch_embeddings.num_patches**0.5)
        grid_h = np.arange(grid_size, dtype = np.float32) * orig_grid_size / grid_size
        grid_w = np.arange(grid_size, dtype = np.float32) * orig_grid_size / grid_size
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis = 0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if add_cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis = 0)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        return pos_embed

    def get_trained_pos_embed(self, embed_dim, grid_size, add_cls_token = False):
        assert embed_dim == 768
        with torch.no_grad():
            resize_fn = transforms.Resize(grid_size)
            orig_pos_embeddings = self.sub_modules["vit_embedding"].position_embeddings.data
            new_embeddings = resize_fn(orig_pos_embeddings[:, 1:, :].permute(0, 2, 1).reshape(1, 768, 14, 14))
            new_embeddings = new_embeddings.reshape(1, 768, grid_size**2).permute(0, 2, 1)
            if add_cls_token:
                new_embeddings = torch.cat((orig_pos_embeddings[:,:1,:], new_embeddings), dim = 1)

        return new_embeddings

    def patch2img(self, x):
        """
        x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
        """
        p = self.config.patch_size
        c = self.config.num_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


if __name__ == "__main__":
    model = PretrainedViTMAE(image_size = 256, patch_size = 16)
    recon_data = model(torch.zeros(32, 3, 256, 256))