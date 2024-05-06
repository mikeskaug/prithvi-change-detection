

from torch import nn
import numpy as np

from config import MODEL_ARGS


class ConvBlock(nn.Module):
    '''This module creates a user-defined number of conv+BN+ReLU layers.
    Args:
        in_channels (int)-- number of input features.
        out_channels (int) -- number of output features.
        kernel_size (int) -- Size of convolution kernel.
        stride (int) -- decides how jumpy kernel moves along the spatial dimensions.
        padding (int) -- how much the input should be padded on the borders with zero.
        dilation (int) -- dilation ratio for enlarging the receptive field.
        num_conv_layers (int) -- Number of conv+BN+ReLU layers in the block.
        drop_rate (float) -- dropout rate at the end of the block.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, num_conv_layers=2, drop_rate=0):
        super(ConvBlock, self).__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation, bias=False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True), ]

        if num_conv_layers > 1:
            if drop_rate > 0:
                layers += [nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, bias=False),
                           nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                           nn.Dropout(drop_rate), ] * (num_conv_layers - 1)
            else:
                layers += [nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, bias=False),
                           nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), ] * (num_conv_layers - 1)

        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.block(inputs)
    

class UpconvBlock(nn.Module):
    '''
    An upsampling block.
    Args:
        in_channels (int) -- number of input features.
        out_channels (int) -- number of output features.
        upmode (str) -- Upsampling method. If "upsample" then a linear upsampling with scale factor
                        of two will be applied using bi-linear as interpolation method.
                        If conv_transpose is chosen then a non-overlapping transposed convolution will
                        be applied to upsample the feature maps.
    '''

    def __init__(self, in_channels, out_channels, upmode='deconv'):
        super(UpconvBlock, self).__init__()

        if upmode == 'upsample':
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            )

        elif upmode == 'conv_transpose':
            self.block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2)

        else:
            raise ValueError('Provided upsampling mode is not recognized.')

    def forward(self, inputs):
        return self.block(inputs)
    

class DamageSegmentation(nn.Module):
    def __init__(self, prithvi_encoder, dropout_rate=0):
        super(DamageSegmentation, self).__init__()

        num_filters = [768, 384, 192, 96, 48] #= [MODEL_ARGS["embed_dim"] // (2**i) for i in range(5)]

        self.encoder = prithvi_encoder # 768 x 64 x 64
        
        
        self.decoder_1 = nn.Sequential(
            UpconvBlock(num_filters[0], num_filters[1], upmode="conv_transpose"),
            ConvBlock(num_filters[1], num_filters[1], num_conv_layers=2, drop_rate=dropout_rate)
        )  # 384 x 128 x 128

        self.decoder_2 = nn.Sequential(
            UpconvBlock(num_filters[1], num_filters[2], upmode="conv_transpose"),
            ConvBlock(num_filters[2], num_filters[2], num_conv_layers=2, drop_rate=dropout_rate)
        )  # 192 x 256 x 256

        self.decoder_3 = nn.Sequential(
            UpconvBlock(num_filters[2], num_filters[3], upmode="conv_transpose"),
             ConvBlock(num_filters[3], num_filters[3], num_conv_layers=2, drop_rate=dropout_rate)
        )  # 96 x 512 x 512

        self.decoder_4 = nn.Sequential(
            UpconvBlock(num_filters[3], num_filters[4], upmode="conv_transpose"),
            ConvBlock(num_filters[4], num_filters[4], num_conv_layers=2, drop_rate=dropout_rate)
        )  # 64 x 1024 x 1024
        

        self.classifier = nn.Conv2d(kernel_size=1, in_channels=num_filters[4], out_channels=MODEL_ARGS['num_classes'])
        
    def forward(self, input):
        embedding = self.encoder(input) # batch size x (64 x 64) x 768 

        # drop cls token
        reshaped_features = embedding[:, 1:, :]

        # reshape
        feature_img_side_length = int(np.sqrt(reshaped_features.shape[1]))
        reshaped_features = reshaped_features.view(-1, feature_img_side_length, feature_img_side_length, MODEL_ARGS["embed_dim"])

        # channels first
        reshaped_features = reshaped_features.permute(0, 3, 1, 2)  # batch size x 768 x 64 x 64

        x = self.decoder_1(reshaped_features) # batch size x 384 x 128 x 128
        x = self.decoder_2(x) # batch size x 192 x 256 x 256
        x = self.decoder_3(x) # batch size x 96 x 512 x 512
        x = self.decoder_4(x) # batch size x 64 x 1024 x 1024

        x = self.classifier(x) # batch size x 4 x 1024 x 1024

        return x