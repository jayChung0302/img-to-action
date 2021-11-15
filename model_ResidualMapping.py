""" Add Codition"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [self.Conv3x3(in_features),
                      nn.InstanceNorm3d(in_features),
                      nn.ReLU(inplace=True),
                      self.Conv3x3(in_features),
                      nn.InstanceNorm3d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def Conv3x3(self, in_features):
        return nn.Conv3d(in_features, in_features, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        return x + self.conv_block(x)


class Residual3dBlock(nn.Module):
    def __init__(self, in_features):
        super(Residual3dBlock, self).__init__()

        conv_block = [self.Conv3x3(in_features),
                      nn.InstanceNorm3d(in_features),
                      nn.ReLU(inplace=True),]

        self.conv_block = nn.Sequential(*conv_block)

    def Conv3x3(self, in_features):
        return nn.Conv3d(in_features, in_features, kernel_size=(2, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        return torch.cat((self.conv_block(x), x), dim=2)


class vid2img(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6):
        super(vid2img, self).__init__()

        # Initial motion block
        model = [nn.Conv3d(input_nc, 16, kernel_size=(3, 3, 3), padding=1),
                 nn.InstanceNorm3d(16),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 16
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv3d(in_features, out_features, kernel_size=(3, 3, 3), stride=2, padding=1),
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2  # 16x56

        # # condense sequences
        model += [nn.Conv3d(in_features, in_features, kernel_size=(16, 1, 1))]  # 1x56

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2  # 1, 128, 1, 56, 56
        for _ in range(2):
            model += [nn.ConvTranspose3d(in_features, out_features, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                         padding=(0, 1, 1), output_padding=(0, 1, 1)),
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        # Output layer
        model += [nn.Conv3d(16, output_nc, 1),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class img2vid(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=5):
        super(img2vid, self).__init__()

        # Initial motion block
        model = [nn.Conv3d(input_nc, 16, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                 nn.InstanceNorm3d(16),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 16
        out_features = in_features * 2
        for _ in range(3):
            model += [nn.Conv3d(in_features, out_features, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  ## 128 x 112 x 112
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True),]

            in_features = out_features  # 128
            out_features = in_features * 2  # 256

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [Residual3dBlock(in_features)]  # residual block, 1,3,7,15,31  # 1x256x31x28x28

        # produce sequences
        model += [nn.ConvTranspose3d(in_features, in_features, kernel_size=(2, 1, 1), stride=1, padding=0)]
        # 1x128x32x28x28
        # Upsampling
        out_features = in_features // 2  # 64
        for _ in range(3):
            model += [nn.ConvTranspose3d(in_features, out_features, kernel_size=(1, 3, 3),
                                         stride=(1, 2, 2), padding=(0,1,1),output_padding=(0,1,1)),
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2  # 1x16x32x224x224

        # Output layer
        model += [nn.Conv3d(16, output_nc, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x, label):
        torch.cat((x, label), dim=1)
        x = self.model(x)
        return x


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return logits

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


class VidD(nn.Module):
    def __init__(self, input_nc, output_nc=1, n_residual_blocks=4):
        super(VidD, self).__init__()

        # Initial motion block
        model = [nn.Conv3d(input_nc, 32, kernel_size=(3, 3, 3), padding=1),
                 nn.InstanceNorm3d(32),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 32
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv3d(in_features, out_features, kernel_size=(3, 3, 3), stride=2, padding=1),
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2  # 16x56

        # condense sequences
        model += [nn.Conv3d(in_features, in_features, kernel_size=(16, 1, 1))]  # 1x56

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        squeeze = [nn.Conv2d(128, output_nc, kernel_size=3),
                   nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
        linear = [nn.Linear(729, 1),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.squeeze = nn.Sequential(*squeeze)
        self.linear = nn.Sequential(*linear)

    def forward(self, x):
        x = self.model(x)
        # print(x.size())
        x = self.squeeze(x.squeeze(2))
        # print(x.size())
        x = self.linear(x.view(x.size(0), -1))
        return x



if __name__ == '__main__':
    gn_vid2img = vid2img(3, 3)
    gn_img2vid = img2vid(3, 3)
    # d_img = models.resnet18()
    d_img = models.mobilenet_v2()
    d_img.classifier[1] = torch.nn.Linear(1280, 1)
    d_vid = VidD(3, 1)

    img = torch.randn(1, 3, 1, 224, 224)
    vid = torch.randn(1, 3, 32, 224, 224)
    # print('network parameters: d_img, d_vid, vid2img, img2vid')
    # print(sum([p.numel() for p in d_img.parameters()]))
    # print(sum([p.numel() for p in d_vid.parameters()]))
    # print(sum([p.numel() for p in gn_vid2img.parameters()]))
    # print(sum([p.numel() for p in gn_img2vid.parameters()]))
    fi = gn_img2vid(img)
    print(fi.size())
    gfi = gn_vid2img(fi)
    #
    gv = gn_vid2img(vid)
    fgv = gn_img2vid(gv)
    print('image cycle')
    print('img size : {}'.format(img.size()))
    print('fi size : {}'.format(fi.size()))
    print('gfi size : {}'.format(gfi.size()))
    print('')
    print('video cycle')
    print('vid size : {}'.format(vid.size()))
    print('gv size : {}'.format(gv.size()))
    print('fgv size : {}'.format(fgv.size()))
    #
    D_vid = VidD(3, 1)
    out = D_vid(vid)
    print(out.size())