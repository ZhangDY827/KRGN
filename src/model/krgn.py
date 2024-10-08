from math import gcd

import torch
import torch.nn as nn
from torch.nn import functional as F
#from model import common
import common

def make_model(args, parent=False):
    return KRGN(args)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'identity':
        return nn.Identity()
    elif act_type == 'relu':
        return nn.ReLU(inplace)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class RecConv(nn.Module):
    """
    --in_dims: number of input channels in the basis kernel
    --out_dims: number of output channels in the basis kernel
    --kernel_size: size of the basis kernel
    --stride: stride of the original convolution
    --padding: padding added to all four sides of the basis kernel
    --groups: groups of the original convolution
    --learn_k: size of the learnable kernel
    """

    def __init__(self, in_dims, out_dims, kernel_size, stride,
                 padding=None, groups=1, learn_k=3):
        super(RefConv, self).__init__()

        assert learn_k <= kernel_size
        self.origin_weight = (out_dims, in_dims // groups,
                                    kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_weight))        
        self.num_2d_kernels = out_dims * in_dims // groups
        G = in_dims * out_dims // (groups ** 2)
        self.kernel_size = kernel_size
        self.conv_transformer = nn.Conv2d(in_dims=self.num_2d_kernels,
                                 out_dims=self.num_2d_kernels,
                                 kernel_size=learn_k, stride=1, padding=learn_k // 2,
                                 groups=G, bias=False)

        nn.init.zeros_(self.conv_transformer.weight)
        self.weight = nn.Parameter(torch.zeros(out_dims), requires_grad=True)
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight +self.conv_transformer(origin_weight).view(*self.origin_weight)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups, bias=self.bias)

# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            activation('lrelu'),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class SRB(nn.Module):
    def __init__(self, in_channels):
        super(SRB, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.act = activation('lrelu')

    def forward(self, x):
        out = self.conv3x3(x) + x
        out = self.act(out)
        return out


class RCG(nn.Module):
    def __init__(self, in_channels, dilation):
        super(RCG, self).__init__()

        groups = 32
        self.num_conv = 3
        print(f'in_channels:{in_channels}, num_conv: {self.num_conv}, groups: {groups}')

        self.conv_acts = []
        for i in range(self.num_conv * 2):
            if i % 2 == 0:
                self.conv_acts.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, 3, 1, 1, 1, groups=groups),
                        #RecConv(in_channels, in_channels, kernel_size=3, stride = 1, padding=None, dilation = dilation, groups=groups, map_k=3),
                        
                        activation('prelu', n_prelu=in_channels)
                    )
                )
            else:
                self.conv_acts.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, 3, 1, dilation, dilation, groups=groups),
                        #RecConv(in_channels, in_channels, kernel_size=3, stride = 1, padding=None, dilation = dilation, groups=groups, map_k=3),
                        activation('prelu', n_prelu=in_channels)
                    )
                )
        self.conv_acts = nn.Sequential(*self.conv_acts)


    def forward(self, x):
        out = x
        for i in range(self.num_conv):
            out = self.conv_acts[i*2](out) + self.conv_acts[i*2+1](out)
        return out + x

class PMRB(nn.Module):
    def __init__(self, in_channels, act, dilation):
        super(PMRB, self).__init__()

        self.num = 3

        self.blocks = [
            RCG(in_channels, dilation) for _ in range(self.num)
        ]
        self.blocks = nn.Sequential(*self.blocks)

        self.conv1x1s = [
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0),
                activation(act)
            ) for _ in range(self.num)
        ]
        self.conv1x1s = nn.Sequential(*self.conv1x1s)

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            activation(act)
        )

        self.conv1x1_act = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0),
            activation('lrelu')
        )

        self.cca = CCALayer(in_channels)
    
    def forward(self, x):
        now = x
        features = []
        for i in range(self.num):
            features.append(self.conv1x1s[i](now))
            now = self.blocks[i](now)
        features.append(self.conv3x3(now))
        features = torch.cat(features, 1)
        out = self.conv1x1_act(features)
        out = self.cca(out)
        return out + x


class KRGN(nn.Module):
    """KRGN network structure.

    Args:
        args.scale (list[int]): Upsampling scale for the input image.
        args.n_colors (int): Channels of the input image.
        args.n_feats (int): Channels of the mid layer.
        args.n_resgroups (int): Number of context feature guided groups.
        args.act (str): Activate function used in network.
        args.rgb_range: 255.
    """
    def __init__(self, args):
        super(KRGN, self).__init__()
        assert len(args.scale) == 1
        scale = args.scale[0]
        n_colors = args.n_colors
        n_feats = args.n_feats
        n_resgroups = args.n_resgroups
        act = args.act
        rgb_range = args.rgb_range
        dilation = int(args.dilation)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)

        self.head_conv = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        self.head_act = activation('identity')

        self.body = []
        for i in range(n_resgroups):
            self.body.append(PMRB(n_feats, act, dilation))
        self.body = nn.Sequential(*self.body)

        self.features_fusion_module = nn.Sequential(
            nn.Conv2d(n_feats * (n_resgroups + 1), n_feats, 1, 1, 0),
            activation('lrelu'),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            activation('identity')
        )

        self.upsampler = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale * scale), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head_conv(x)

        now = self.head_act(x)
        outs = [now]
        for main_block in self.body:
            now = main_block(now)
            outs.append(now)

        outs = torch.cat(outs, 1)
        y = self.features_fusion_module(outs) + x

        y = self.upsampler(y)
        y = self.add_mean(y)

        return y


if __name__ == '__main__':
    # test network
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    import argparse
    args = argparse.Namespace()
    # args.scale = [2]
    # args.patch_size = 128
    # args.n_colors = 3
    # args.n_feats = 64
    # args.n_resgroups = 8
    # args.act = 'lrelu'
    # args.rgb_range = 255
    # args.block_type = 'cfgg'

    args.scale = [4]
    args.patch_size = 128
    args.n_colors = 3
    args.n_feats = 64
    args.n_resgroups = 9#9
    args.act = 'identity'
    args.rgb_range = 255
    args.dilation = 3
    # import pdb
    # pdb.set_trace()
    model = KRGN(args)
    model.eval()
    from thop import profile
    inp = torch.randn(1,3,64,64)
    flops, params = profile(model, inputs=(inp,))
    #print(flops)
    print(params)

    #inp = torch.randn(1,3,180,320)
    #inp = inp.cuda()
    #model = model.cuda()
    #for i in range(100):
    #    model(inp)

    # from torchsummary import summary
    # summary(model.cuda(), input_size=(3, 720 // 4, 1280 // 4), batch_size=1)


    from torchinfo import summary

    summary(model, input_size=(1, 3, 180, 320), col_names=["num_params", "mult_adds"])
