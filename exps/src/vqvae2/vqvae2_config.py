

class VQVAE2Config():
    def __init__(self, image_size, patch_size, n_clusters = 256, num_skipped_scales = 1):
        self.share = image_size//patch_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.in_channel = 3
        self.channel = 128
        self.n_res_block = 2
        self.n_res_channel = 32
        self.embed_dim = 64
        self.n_embed = n_clusters
        self.decay = 0.99

        self.num_skipped_scales = num_skipped_scales

        self.lr = 3e-4

        self.latent_loss_weight = 0.25

        self.independent_decoder = False

        self.ll_ratio = 1e-4

        self.out_dim = 30
