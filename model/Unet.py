import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_conv(nn.Module):
    """ conv_conv: (conv[3*3] + BN + ReLU) *2 """

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(conv_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        X = self.conv(X)
        return X


class downconv(nn.Module):
    """ downconv: conv_conv => maxpool[2*2] """

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(downconv, self).__init__()
        self.conv = conv_conv(in_channels, out_channels, bn_momentum)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        X = self.conv(X)
        pool_X = self.pool(X)
        return pool_X, X


class upconv_concat(nn.Module):
    """ upconv_concat: upconv[2*2] => cat => conv_conv """

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(upconv_concat, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = conv_conv(in_channels, out_channels, bn_momentum)

    def forward(self, X1, X2):
        X1 = self.upconv(X1)
        feature_map = torch.cat((X2, X1), dim=1)
        X1 = self.conv(feature_map)
        return X1


class UNet(nn.Module):
    """ UNet(4-level): Encoder[downconv *4 => conv_conv]=> Decoder[upconv *4 => conv[1*1]]"""

    def __init__(self, in_channels, out_channels, proj_dim=32, starting_filters=32, bn_momentum=0.1, projection_head=True):
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = downconv(in_channels, starting_filters, bn_momentum)
        self.conv2 = downconv(starting_filters, starting_filters * 2, bn_momentum)
        self.conv3 = downconv(starting_filters * 2, starting_filters * 4, bn_momentum)
        self.conv4 = downconv(starting_filters * 4, starting_filters * 8, bn_momentum)
        self.conv5 = conv_conv(starting_filters * 8, starting_filters * 16, bn_momentum)
        # Decoder
        self.upconv1 = upconv_concat(starting_filters * 16, starting_filters * 8, bn_momentum)
        self.upconv2 = upconv_concat(starting_filters * 8, starting_filters * 4, bn_momentum)
        self.upconv3 = upconv_concat(starting_filters * 4, starting_filters * 2, bn_momentum)
        self.upconv4 = upconv_concat(starting_filters * 2, starting_filters, bn_momentum)
        self.conv_out = nn.Conv2d(starting_filters, out_channels, kernel_size=1, padding=0, stride=1)
        # projection head
        if projection_head:
            # projection head
            self.projhead = nn.Sequential(
                nn.Conv2d(starting_filters * 16, starting_filters * 8, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm2d(256),
                nn.Conv2d(in_channels=256, out_channels=proj_dim, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.projhead=None

        self._init_weight()

    def forward(self, X):
        X, conv1 = self.conv1(X)
        X, conv2 = self.conv2(X)
        X, conv3 = self.conv3(X)
        X, conv4 = self.conv4(X)

        X = self.conv5(X)

        if self.projhead is not None:
            featmap = self.projhead(X)
        else:
            featmap = None

        X = self.upconv1(X, conv4)
        X = self.upconv2(X, conv3)
        X = self.upconv3(X, conv2)
        X = self.upconv4(X, conv1)

        X = self.conv_out(X)
        if featmap is not None:
            featmap = F.interpolate(featmap, size=X.size()[2:], mode='bilinear', align_corners=True)
            return featmap, X
        else:
            return X

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.projhead is not None:
            for param in self.projhead.parameters():
                param.requires_grad = False  # NOT updated by gradient

if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=6, proj_dim=64)

    img = torch.randn(2, 3, 512, 512)

    x = model(img)

    # print(f.size())
    print(x.size())
