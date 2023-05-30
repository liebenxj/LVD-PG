import torch
import torch.nn as nn
from layers import *
from model import *
from utils import *


class ContextualPixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, 
                 resnet_nonlinearity='concat_elu', input_channels=3, 
                 context_channels=256, n_downscale=2, n_levels=3):
        super(ContextualPixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' : 
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else : 
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.feature_enc = nn.Conv2d(input_channels + 1, nr_filters, kernel_size = 1, padding = 0)

        self.n_downscale = n_downscale
        self.n_levels = n_levels

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] * (n_levels - n_downscale) + [nr_resnet + 1] * n_downscale
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters, 
                                                self.resnet_nonlinearity, additional_input_size = context_channels) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters, 
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters, 
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters, 
                                                    nr_filters, stride=(2,2)) for _ in range(2)])
        
        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters, 
                                                    stride=(2,2)) for _ in range(2)])
        
        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters, 
                                                    nr_filters, stride=(2,2)) for _ in range(2)])
        
        self.u_init = down_shifted_conv2d(nr_filters + context_channels, nr_filters, filter_size=(2,3), 
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(nr_filters + context_channels, nr_filters, 
                                            filter_size=(1,3), shift_output_down=True), 
                                       down_right_shifted_conv2d(nr_filters + context_channels, nr_filters, 
                                            filter_size=(2,1), shift_output_right=True)])
    
        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def encode_x(self, x, context):
        xf = self.feature_enc(x)
        h, w = xf.size(2), xf.size(3)
        cf = context[:,:,None,None].repeat(1, 1, h, w)
        return torch.cat((xf, cf), dim = 1)

    def forward(self, x, context, sample=False):
        # similar as done in the tf repo :  
        if self.init_padding is None and not sample: 
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(1, 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.to(x.device) if x.is_cuda else padding
        
        if sample : 
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(1, 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.to(x.device) if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding.repeat(x.size(0), 1, 1, 1)), 1) # add a channel with ones
        x = self.encode_x(x, context)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(self.n_levels):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            if i < self.n_downscale: 
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()
        
        for i in range(self.n_levels):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list, context = context)

            # upscale (only twice)
            if i < self.n_downscale:
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out