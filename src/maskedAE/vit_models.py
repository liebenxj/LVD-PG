import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer
from vector_quantize_pytorch import VectorQuantize


class PassThrough(nn.Module):
    def __init__(self):
        super(PassThrough, self).__init__()

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        if not output_attentions:
            return (hidden_states,)
        else:
            return hidden_states, None


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, image_size=32, patch_size=2, num_channels=3, embed_dim=64):
        super(PatchEmbeddings, self).__init__()

        image_size = self.to_2tuple(image_size)
        patch_size = self.to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        batch_size, num_channels, height, width = pixel_values.shape
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
                )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x

    @staticmethod
    def to_2tuple(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return (x, x)


class MAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.register_buffer(
            "position_embeddings",
            self.get_sinusoid_encoding_table(num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    @staticmethod
    def get_sinusoid_encoding_table(n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def interpolate_pos_encoding(self, embeddings, height, width):
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        npatch = embeddings.shape[1] - 1
        N = self.position_embeddings.shape[1] - 1
        if npatch == N and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if not self.config.homogeneous:
            if interpolate_pos_encoding:
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                embeddings = embeddings + self.position_embeddings
        
        embeddings = self.dropout(embeddings)

        return embeddings


class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super(ViTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attn_heads != 0 :
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attn_heads}."
            )

        self.num_attn_heads = config.num_attn_heads
        self.attention_head_size = config.hidden_size // config.num_attn_heads
        self.all_head_size = self.num_attn_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # Original size of x: [B, L, hidden_size]
        new_x_shape = x.size()[:-1] + (self.num_attn_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_back_from_scores(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = x.size()[:-2] + (self.all_head_size,)
        x = x.view(*new_context_layer_shape)
        return x

    def forward(self, hidden_states, head_mask=None, output_attentions=False):

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = self.transpose_back_from_scores(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        return outputs


class ViTFC(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config, interm_input = False):
        super(ViTFC, self).__init__()
        if interm_input:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self, config):
        super(ViTAttention, self).__init__()

        self.attention = ViTSelfAttention(config)
        self.output = ViTFC(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ViTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTLayer(nn.Module):
    def __init__(self, config):
        super(ViTLayer, self).__init__()

        self.attention = ViTAttention(config) if not config.homogeneous else PassThrough()
        self.intermediate = ViTIntermediate(config)
        self.output = ViTFC(config, interm_input = True)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        
        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        
        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        
        outputs = (layer_output,) + outputs

        return outputs


class ViTEncoder(nn.Module):
    def __init__(self, config):
        super(ViTEncoder, self).__init__()

        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, head_mask=None, output_attentions=False, output_hidden_states=False):

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state = hidden_states,
            hidden_states = all_hidden_states,
            attentions = all_self_attentions,
        )


class MAEPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the average of hidden states
        pooled_output = self.layernorm(hidden_states.mean(1))
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MAEEncoder(nn.Module):
    def __init__(self, config):
        super(MAEEncoder, self).__init__()
        self.config = config

        self.embeddings = MAEEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = MAEPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        self.apply(_init_weights)

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, pixel_values=None, patch_masks=None, attention_mask=None, head_mask=None,
                output_attentions=False, output_hidden_states=False, interpolate_pos_encoding=None):

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        
        # MAE: Skip [CLS] and sample a subset of patches
        embedding_output = embedding_output[:, 1:]
        if patch_masks is not None:
            batch_size, _, dim = embedding_output.shape
            if patch_masks.dim() == 1:
                embedding_output = embedding_output[:,~patch_masks,:].reshape(batch_size, -1, dim)
            elif patch_masks.dim() == 2:
                embedding_output = embedding_output[~patch_masks,:].reshape(batch_size, -1, dim)
            else:
                raise ValueError(f"Does not support `patch_masks` with size {patch_masks.size()}")

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_head_mask(self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility

        return head_mask


class MAEDecoder(ViTEncoder):
    pass


class MaskedAE(nn.Module):
    def __init__(self, enc_config, dec_config, mask_generator, vec_quantizer_config = None):
        super(MaskedAE, self).__init__()

        self.enc_config = enc_config
        self.vec_quantizer_config = vec_quantizer_config

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

        if self.vec_quantizer_config is not None:
            self.vq = VectorQuantize(
                dim = self.enc_config.hidden_size,
                codebook_size = self.vec_quantizer_config.codebook_size,
                codebook_dim = self.vec_quantizer_config.codebook_dim,
                use_cosine_sim = self.vec_quantizer_config.use_cosine_sim, 
                threshold_ema_dead_code = self.vec_quantizer_config.threshold_ema_dead_code
            )

        self.device = torch.device("cpu")

    def to(self, device):
        super(MaskedAE, self).to(device)

        self.device = device

    def forward(self, imgs, patch_masks=None, head_mask=None, output_attentions=False,
                output_hidden_states=False, interpolate_pos_encoding=False):

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
        if self.vec_quantizer_config is not None:
            sequence_output, _, commit_loss = self.vq(sequence_output)

        sequence_output = self.encoder_to_decoder(sequence_output)
        batch_size, unmask_size, dim = sequence_output.shape

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

        decoder_inputs = torch.cat([sequence_output + pos_emd_unmask, self.mask_token + pos_emd_mask], dim=1)

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
        if self.vec_quantizer_config is not None:
            loss += self.vec_quantizer_config.commit_loss_weight * commit_loss.mean()

        return SequenceClassifierOutput(
            loss=loss,
            logits=recon_img,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_patch_features(self, imgs, patch_masks):
        outputs = self.vit(imgs, patch_masks = patch_masks)
        sequence_output = outputs[0] # (B, num_unmasked_patches, feature_dim)
        return sequence_output

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


class RandomMaskingGenerator():
    def __init__(self, config):
        input_size = config.image_size // config.patch_size
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size
        self.num_patches = self.height * self.width

        self.min_mask_ratio = config.min_mask_ratio
        self.max_mask_ratio = config.max_mask_ratio

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(self.num_patches, self.num_mask)
        return repr_str

    def __call__(self, mask_ratio = None):
        if mask_ratio is None:
            mask_ratio = np.random.random() * (self.max_mask_ratio - self.min_mask_ratio) + self.min_mask_ratio
        num_mask = int(mask_ratio * self.num_patches)
        num_mask = max(num_mask, 1)

        mask = np.hstack([
            np.zeros(self.num_patches - num_mask),
            np.ones(num_mask),
        ]).astype(bool)
        np.random.shuffle(mask)
        return mask


if __name__ == "__main__":
    from config import *

    enc_config = MAE_ViT_encoder_config()
    dec_config = MAE_ViT_decoder_config()

    mask_generator = RandomMaskingGenerator(enc_config)
    model = MaskedAE(enc_config, dec_config, mask_generator)

    optimizer, scheduler = get_optimizer_and_scheduler(model, 100000)

    y = model(torch.zeros(64, 3, 32, 32))