from distutils.command.config import config
import torch
from torch import nn
from torch.nn import functional as F
from utils import *
from einops import rearrange

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                sane_index_shape=True, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[1], z_q.shape[2])

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q




class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, lls=False):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            if not lls:
                blocks.extend(
                    [
                        nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
                    ]
                )
            else:
                blocks.extend(
                    [
                        nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(channel // 2, 30, 4, stride=2, padding=1),
                    ]
                )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class IndependentDecoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()
        self.discret_feats = 30
        self.net = nn.Sequential(
            nn.Linear(in_channel, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4 * 4 * self.discret_feats)
        )


    def forward(self, input):
        B, F, H, W = input.size()
        x = input.permute(0, 2, 3, 1).reshape(B * H * W, F)
        x = self.net(x)
        return x.reshape(B, H, W, self.discret_feats, 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(B, self.discret_feats, H * 4, W * 4)


class VQVAE2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        in_channel = self.config.in_channel
        channel = self.config.channel
        n_res_block = self.config.n_res_block
        n_res_channel = self.config.n_res_channel
        embed_dim = self.config.embed_dim
        n_embed = self.config.n_embed
        beta = self.config.latent_loss_weight

        if self.config.patch_size == 4:
            stride1, stride2 = 4, 2
        else:
            raise NotImplementedError("Unsupported stride")

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=stride1)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=stride2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)

        self.quantize_t = VectorQuantizer(n_embed, embed_dim, beta, legacy=False)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=stride2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = VectorQuantizer(n_embed, embed_dim, beta, legacy=False)

        if not self.config.independent_decoder:
            self.upsample_t = nn.ConvTranspose2d(
                embed_dim, embed_dim, 4, stride=2, padding=1
            )
            self.dec = Decoder(
                embed_dim + embed_dim,
                in_channel,
                channel,
                n_res_block,
                n_res_channel,
                stride=4,
                lls = True
            )

        else:
            self.dec = IndependentDecoder(
                embed_dim + embed_dim,
                in_channel,
                channel,
                n_res_block,
                n_res_channel,
                stride=4,
            )

        self.mse_loss = nn.MSELoss()
    


    def forward(self, input, input_ll):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        # recon_loss = self.mse_loss(dec, input)
        lls = discretized_mix_logistic_lls(input_ll,dec)
        ll_loss = discretized_mix_logistic_loss(input_ll,dec)
        latent_loss = diff.mean()

        # loss = recon_loss + self.config.latent_loss_weight * latent_loss
        loss = ll_loss * self.config.ll_ratio + latent_loss     
        
        return {
            "x_recon": dec, 
            # "recon_loss": recon_loss,
            "ll_loss": ll_loss,
            "lls": lls,
            "latent_loss": latent_loss,
            "loss": loss
        }

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b


    def decode(self, quant_t, quant_b):
        if self.config.independent_decoder:
            repeats = 2*torch.ones(size=(4,),dtype=torch.int).to(quant_t.device).long()
            upsample_t = quant_t.repeat_interleave(repeats,dim=2)
            upsample_t = upsample_t.repeat_interleave(repeats,dim=3)
        else:
            upsample_t = self.upsample_t(quant_t)

        quant = torch.cat([upsample_t, quant_b], 1)

        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

    def get_patch_features(self, input):
        _, _, _, id_t, id_b = self.encode(input)
        return id_b

    def get_continous_features(self,input):
        quant_t, quant_b, _, _, _ = self.encode(input)
        if self.config.independent_decoder:
            repeats = 2*torch.ones(size=(4,),dtype=torch.int).to(quant_t.device).long()
            upsample_t = quant_t.repeat_interleave(repeats,dim=2)
            upsample_t = upsample_t.repeat_interleave(repeats,dim=3)
        else:
            upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        return quant