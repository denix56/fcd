import numpy as np
import torch
import torch.nn as nn

def lrelu(inplace=False):
    return nn.LeakyReLU(0.01, inplace)

def get_activation(activation):
    if activation == 'relu':
        act_class = nn.ReLU
    elif activation == 'lrelu':
        act_class = lrelu
    elif activation == 'silu':
        act_class = nn.SiLU
    elif activation == 'mish':
        act_class = nn.Mish
    else:
        raise NotImplementedError()
    return act_class


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, act_class):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            act_class(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, num_channels=3, activation='relu'):
        super(Generator, self).__init__()
        
        act_class = get_activation(activation)
        
        layers = []
        layers.append(nn.Conv2d(num_channels+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(act_class(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(act_class(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, act_class=act_class))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(act_class(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, num_channels, kernel_size=7, stride=1, padding=3, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        inputs = torch.cat([x, c], dim=1)

        delta = self.main(inputs)
        x_fake = torch.tanh(delta + x)

        # Set the invalid pixels in original image as invalid in the generated image.
        invalid_mask = (x == -1).float()
        x_fake = (x_fake * (1 - invalid_mask)) - invalid_mask

        return x_fake


class AttnLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.15, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, num_channels=3, activation='lrelu',
                 n_feature_layers=4, interm_non_act=False, use_attention=False):
        super(Discriminator, self).__init__()
        
        act_class = get_activation(activation)

        if interm_non_act:
            create_layer = lambda in_c, out_c, kernel_size, stride, padding, is_first: \
                nn.Sequential(nn.Identity() if is_first else act_class(inplace=True),
                              nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding))
        else:
            create_layer = lambda in_c, out_c, kernel_size, stride, padding, is_first: \
                nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
                              act_class(inplace=True))
        
        layers = []
        layers.append(create_layer(num_channels, conv_dim, kernel_size=4, stride=2, padding=1, is_first=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if (i+1) % 3 == 0:
                layers.append(AttnLayer(curr_dim))
            layers.append(create_layer(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, is_first=False))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.features = nn.ModuleList(layers[:n_feature_layers])
        if len(layers[n_feature_layers:]) > 0:
            self.main = nn.Sequential(*layers[n_feature_layers:])
        else:
            self.main = nn.Identity()
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        features = [self.features[0](x)]
        for l in self.features[1:]:
            features.append(l(features[-1]))
        h = self.main(features[-1])
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1)), [features[-1]]

