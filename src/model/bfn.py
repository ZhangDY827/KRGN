from math import gcd

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import common


def make_model(args, parent=False):
    return BFN(args)

def generate_masks(num):
    masks = []
    for i in range(num):
        now = list(range(2 ** num))
        length = 2 ** (num - i)
        for j in range(2 ** i):
            tmp = now[j*length:j*length+length//2]
            now[j*length:j*length+length//2] = now[j*length+length//2:j*length+length]
            now[j*length+length//2:j*length+length] = tmp
        masks.append(now)
    return masks


class CA(nn.Module):
    def __init__(self, in_channels, reduction, act, skip_connection):
        super(CA, self).__init__()
        assert reduction >= 1 and in_channels >= reduction
        self.skip_connection = skip_connection
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2 // reduction, 1),
            act(in_channels * 2 // reduction),
            nn.Conv2d(in_channels * 2 // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        z = torch.mean(torch.mean((x - y) ** 2, dim=2, keepdim=True), dim=3, keepdim=True) ** 0.5
        a = torch.cat([y, z], 1)
        a = self.fc(a)
        if self.skip_connection:
            return x * a + x
        else:
            return x * a


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, in_channels, act, skip_connection):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.act1 = act(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.ca = CA(in_channels, 4, act, skip_connection)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.ca(y)
        y += x
        return y


class ButterflyConv_v1(nn.Module):
    def __init__(self, in_channels, out_channels, act, dilation, skip_connection):
        super(ButterflyConv_v1, self).__init__()

        min_channels = min(in_channels, out_channels)
        assert (min_channels & (min_channels - 1)) == 0 # Is min_channels = 2^n?

        if in_channels == out_channels:
            self.head = nn.Identity()
            self.tail = nn.Identity()
        elif in_channels > out_channels:
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation, groups=gcd(in_channels, out_channels)),
                act(out_channels)
            )
            self.tail = nn.Identity()
        elif in_channels < out_channels:
            self.head = nn.Identity()
            self.tail = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation, groups=gcd(in_channels, out_channels)),
                act(out_channels)
            )
        else:
            raise NotImplementedError("")

        self.num_butterflies = 0
        for i in range(10000):
            if 2 ** i == min_channels:
                self.num_butterflies = i
                break
        self.masks = generate_masks(self.num_butterflies)

        self.skip_connection = skip_connection

        self.conv_acts = []
        for i in range(self.num_butterflies * 2):
            self.conv_acts.append(
                nn.Sequential(nn.Conv2d(min_channels, min_channels, 3, 1, dilation, dilation, groups=min_channels), act(min_channels))
            )
        self.conv_acts = nn.Sequential(*self.conv_acts)

    def forward(self, x):
        x = self.head(x)

        last = x
        for i in range(self.num_butterflies):
            shuffled_last = last[:,self.masks[i],:,:]
            now = self.conv_acts[i*2](last) + self.conv_acts[i*2+1](shuffled_last)
            if self.skip_connection:
                now = now + last
            last = now
        now = now + x

        now = self.tail(now)
        return now


class ButterflyConv_v2(nn.Module):
    def __init__(self, in_channels, out_channels, act, dilation, skip_connection):
        super(ButterflyConv_v2, self).__init__()

        min_channels = min(in_channels, out_channels)
        assert (min_channels & (min_channels - 1)) == 0 # Is min_channels = 2^n?
        
        if in_channels == out_channels:
            self.head = nn.Identity()
            self.tail = nn.Identity()
        elif in_channels > out_channels:
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation, groups=gcd(in_channels, out_channels)),
                act(out_channels)
            )
            self.tail = nn.Identity()
        elif in_channels < out_channels:
            self.head = nn.Identity()
            self.tail = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation, groups=gcd(in_channels, out_channels)),
                act(out_channels)
            )
        else:
            raise NotImplementedError("")

        self.num_butterflies = 0
        for i in range(10000):
            if 2 ** i == min_channels:
                self.num_butterflies = i
                break
        self.masks = generate_masks(self.num_butterflies)

        self.skip_connection = skip_connection

        self.convs = []
        for i in range(self.num_butterflies * 2):
            self.convs.append(nn.Conv2d(min_channels, min_channels, 3, 1, dilation, dilation, groups=min_channels))
        self.convs = nn.Sequential(*self.convs)

        self.acts = []
        for i in range(self.num_butterflies):
            self.acts.append(act(min_channels))
        self.acts = nn.Sequential(*self.acts)

    def forward(self, x):
        x = self.head(x)

        last = x
        for i in range(self.num_butterflies):
            shuffled_last = last[:,self.masks[i],:,:]
            now = self.convs[i*2](last) + self.convs[i*2+1](shuffled_last)
            if self.skip_connection:
                now = now + last
            now = self.acts[i](now)
            last = now
        now = now + x

        now = self.tail(now)
        return now


class ButterflyConv_v3(nn.Module):
    def __init__(self, in_channels, out_channels, act, dilation, skip_connection):
        super(ButterflyConv_v3, self).__init__()

        min_channels = min(in_channels, out_channels)
        assert (min_channels & (min_channels - 1)) == 0 # Is min_channels = 2^n?

        if in_channels == out_channels:
            self.head = nn.Identity()
            self.tail = nn.Identity()
        elif in_channels > out_channels:
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation, groups=gcd(in_channels, out_channels)),
                act(out_channels)
            )
            self.tail = nn.Identity()
        elif in_channels < out_channels:
            self.head = nn.Identity()
            self.tail = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation, groups=gcd(in_channels, out_channels)),
                act(out_channels)
            )
        else:
            raise NotImplementedError("")

        self.num_butterflies = 0
        for i in range(10000):
            if 2 ** i == min_channels:
                self.num_butterflies = i
                break
        self.masks = generate_masks(self.num_butterflies)

        self.skip_connection = skip_connection

        self.convs = []
        for i in range(self.num_butterflies * 2):
            self.convs.append(nn.Conv2d(min_channels, min_channels, 3, 1, dilation, dilation, groups=min_channels))
        self.convs = nn.Sequential(*self.convs)

        self.act = act(min_channels)

    def forward(self, x):
        x = self.head(x)

        last = x
        for i in range(self.num_butterflies):
            shuffled_last = last[:,self.masks[i],:,:]
            now = self.convs[i*2](last) + self.convs[i*2+1](shuffled_last)
            if self.skip_connection:
                now = now + last
            last = now
        now = self.act(now)
        now = now + x
        
        now = self.tail(now)
        return now


class MainBlock_v1(nn.Module):
    def __init__(self, in_channels, act, butterfly_conv, skip_connection):
        super(MainBlock_v1, self).__init__()
        self.butterfly_conv = butterfly_conv(in_channels, in_channels, act, 1, skip_connection)

    def forward(self, x):
        return self.butterfly_conv(x)


class MainBlock_v2(nn.Module):
    def __init__(self, in_channels, act, butterfly_conv, skip_connection):
        super(MainBlock_v2, self).__init__()
        self.butterfly_conv = butterfly_conv(in_channels, in_channels, act, 1, skip_connection)
        self.butterfly_dconv = butterfly_conv(in_channels, in_channels, act, 2, skip_connection)
        self.butterfly_final_conv = butterfly_conv(in_channels * 2, in_channels, act, 1, skip_connection)

    def forward(self, x):
        x1 = self.butterfly_conv(x)
        x2 = self.butterfly_dconv(x)
        out = torch.cat([x1, x2], 1)
        out = self.butterfly_final_conv(out)
        return out + x


class MainBlock_v3(nn.Module):
    def __init__(self, in_channels, act, butterfly_conv, skip_connection):
        super(MainBlock_v3, self).__init__()
        self.butterfly_conv = butterfly_conv(in_channels, in_channels, act, 1, skip_connection)
        self.butterfly_dconv = butterfly_conv(in_channels, in_channels, act, 2, skip_connection)
        self.butterfly_final_conv = butterfly_conv(in_channels * 2, in_channels, act, 1, skip_connection)
        self.ca = CA(in_channels, 4, act, skip_connection)

    def forward(self, x):
        x1 = self.butterfly_conv(x)
        x2 = self.butterfly_dconv(x)
        out = torch.cat([x1, x2], 1)
        out = self.butterfly_final_conv(out)
        out = self.ca(out)
        return out + x


class BFN(nn.Module):
    """BFN network structure.

    Args:
        args.scale (list[int]): Upsampling scale for the input image.
        args.n_colors (int): Channels of the input image.
        args.n_feats (int): Channels of the mid layer.
        args.n_resblocks (int): Number of main blocks.
        args.act (str): Activate function used in BFN. Default: nn.PReLU.
        args.rgb_range: .
        args.main_block_version:
        args.butterfly_conv_version:
        args.skip_connection (bool):.
    """
    def __init__(self, args):
        super(BFN, self).__init__()
        assert len(args.scale) == 1
        scale = args.scale[0]
        n_colors = args.n_colors
        n_feats = args.n_feats
        n_resblocks = args.n_resblocks
        if args.act == 'relu':
            act = nn.ReLU
        elif args.act == 'prelu':
            act = nn.PReLU
        else:
            raise NotImplementedError("")
        rgb_range = args.rgb_range

        if args.main_block_version == 'v1':
            main_block = MainBlock_v1
        elif args.main_block_version == 'v2':
            main_block = MainBlock_v2
        elif args.main_block_version == 'v3':
            main_block = MainBlock_v3
        else:
            raise NotImplementedError("")

        if args.butterfly_conv_version == 'v1':
            butterfly_conv = ButterflyConv_v1
        elif args.butterfly_conv_version == 'v2':
            butterfly_conv = ButterflyConv_v2
        elif args.butterfly_conv_version == 'v3':
            butterfly_conv = ButterflyConv_v3
        else:
            raise NotImplementedError("")

        skip_connection = args.skip_connection

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)

        self.head = nn.Sequential(nn.Conv2d(n_colors, n_feats, 3, 1, 1), act())

        self.main_blocks = []
        for i in range(n_resblocks):
            self.main_blocks.append(main_block(n_feats, act, butterfly_conv, skip_connection))
        self.main_blocks = nn.Sequential(*self.main_blocks)

        self.features_fusion_module = nn.Conv2d(n_feats * (n_resblocks + 1), n_feats, 3, 1, 1)

        self.upsampler = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale * scale), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

        # self.features_fusion_module = nn.Sequential(
        #     butterfly_conv(n_feats * (n_resblocks + 1), n_feats * 4, act, 1, skip_connection),
        #     butterfly_conv(n_feats * 4, n_feats * 2, act, 1, skip_connection),
        #     # ResBlock(n_feats * 2, act),
        #     butterfly_conv(n_feats * 2, n_feats, act, 1, skip_connection),
        #     RCAB(n_feats, act, skip_connection),
        # )

        # self.upsampler = nn.Sequential(
        #     nn.Conv2d(n_feats, n_feats // 4 * (scale * scale), 3, 1, 1),
        #     nn.PixelShuffle(scale),
        #     act(n_feats // 4),
        # )

        # self.tail = RCAB(n_feats // 4, act, skip_connection)

        # self.final_conv = nn.Conv2d(n_feats // 4, n_colors, 3, 1, 1)

        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        now = x
        outs = [now]
        for main_block in self.main_blocks:
            now = main_block(now)
            outs.append(now)

        out = torch.cat(outs, 1)
        out = self.features_fusion_module(out) + x

        out = self.upsampler(out)

        # out = self.tail(out)

        # out = self.final_conv(out)

        out = self.add_mean(out)

        return out


if __name__ == '__main__':
    # test network
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    import argparse
    args = argparse.Namespace()
    args.scale = [4]
    args.patch_size = 192
    args.n_colors = 3
    args.n_feats = 64
    args.n_resblocks = 9
    args.act = 'prelu'
    args.rgb_range = 255
    args.main_block_version = 'v2'
    args.butterfly_conv_version = 'v1'
    args.skip_connection = False

    # import pdb
    # pdb.set_trace()
    model = BFN(args)
    model.eval()

    from torchsummaryX import summary
    x = summary(model.cuda(), torch.zeros((1, 3, 720 // 4, 1280 // 4)).cuda())
    # import pdb
    # pdb.set_trace()

    # from torchsummary import summary
    # summary(model.cuda(), input_size=(3, 48, 48), batch_size=1)

    # x = torch.randn(1, 3, 48, 48)
    # flops, params = profile(model, (x,))
    # print('train_flops: ', flops, 'train_params: ', params)

    # model.eval()

    # x = torch.randn(1, 3, 48, 48)
    # flops, params = profile(model, (x,))
    # print('eval_flops: ', flops, 'eval_params: ', params)




    # 300*(batch_size*1000)/batch_size=300000 次迭代
    # 设每次迭代需要x秒，那么训练完毕需要300000x秒，折合83.3333x小时
    # x的可接受范围在0.5s左右，也即每次迭代必须在0.5s左右结束
