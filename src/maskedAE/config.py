import torch
import torch.nn as nn
import torch.optim as optim


class MAE_ViT_encoder_config():
    def __init__(self, image_size = 32, patch_size = 2, num_channels = 3, large_model = False):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.qkv_bias = True

        self.attention_probs_dropout_prob = 0.0
        self.hidden_dropout_prob = 0.0
        self.hidden_act = "gelu"
        self.layer_norm_eps = 1e-12
        self.initializer_range = 0.02

        self.homogeneous = False

        if large_model:
            self.hidden_size = 512
            self.num_attn_heads = 8
            self.intermediate_size = 1024
            self.num_hidden_layers = 8
        else:
            self.hidden_size = 256
            self.num_attn_heads = 8
            self.intermediate_size = 512
            self.num_hidden_layers = 6

        self.min_mask_ratio = 0.1
        self.max_mask_ratio = 0.9


class MAE_ViT_decoder_config(MAE_ViT_encoder_config):
    def __init__(self, image_size = 32, patch_size = 2, num_channels = 3, large_model = False):
        super(MAE_ViT_decoder_config, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        if large_model:
            self.num_hidden_layers = 6
        else:
            self.num_hidden_layers = 4


class MAE_ViT_quantizer_config():
    def __init__(self, codebook_size = 512):
        self.codebook_size = codebook_size

        self.codebook_dim = 32
        self.use_cosine_sim = False
        self.threshold_ema_dead_code = 2

        self.commit_loss_weight = 0.25


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

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr = 5e-5)

    def lr_lambda(current_step):
        num_warmup_steps = 400
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return max(0.1, float(num_training_steps - current_step) / float(num_training_steps - num_warmup_steps))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler