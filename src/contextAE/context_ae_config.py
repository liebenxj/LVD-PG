import torch
import torch.nn as nn
import torch.optim as optim


class MAE_ViT_encoder_config():
    def __init__(self, image_size = 32, patch_size = 2, num_channels = 3):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.qkv_bias = True

        self.attention_probs_dropout_prob = 0.0
        self.hidden_dropout_prob = 0.0
        self.hidden_act = "gelu"
        self.layer_norm_eps = 1e-12
        self.initializer_range = 0.02

        self.hidden_size = 256
        self.num_attn_heads = 8
        self.intermediate_size = 512
        self.num_hidden_layers = 6


class PixelCNN_decoder_config():
    def __init__(self, image_size = 32, patch_size = 2, num_channels = 3, num_levels = 3):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        
        assert patch_size * 2**(num_levels-1) <= image_size, f"Too many levels ({num_levels}) for images of size {image_size}."
        self.num_levels = num_levels

        self.num_res_layers = 5
        self.hidden_size = 160
        self.num_logistic_mix = 10
        self.context_size = 128

        self.position_encoding_dim = 64

        self.discritize_features = False
        self.n_clusters = 0


class ContextualPixelCNN_config():
    def __init__(self, image_size = 32, patch_size = 8, num_channels = 3):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.num_res_layers = 5
        self.hidden_size = 160
        self.num_logistic_mix = 10

        self.position_encoding_dim = 64

        self.densenet_ninit_features = 32
        self.densenet_growth_rate = 16
        self.densenet_nblocks = 2
        self.densenet_block_size = 8
        self.densenet_compression = 0.5

        self.context_dim = 512

        self.descretize_features = False
        self.n_clusters = 0


def get_optimizer_and_scheduler(model, num_training_steps):

    def get_parameter_names(model, forbidden_layer_types):
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        result += list(model._parameters.keys())
        return result

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0
        }
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr = 1e-4)

    def lr_lambda(current_step):
        num_warmup_steps = 400
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return max(0.04, float(num_training_steps - current_step) / float(num_training_steps - num_warmup_steps))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler
