# UPDATED ON JULY 26TH
# @ WANGSHIROU
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


def load_pretrained_model(model, model_name):
    state_dict = model.state_dict()
    model_dict = {}
    pretrain_dict = model_zoo.load_url(model_urls[model_name])

    for k, v in pretrain_dict.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    return model


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=3, output_stride=16, bn_momentum=0.1):
        if output_stride == 16:
            dilations = [1, 1, 1, 2]
            strides = [1, 2, 2, 1]
        elif output_stride == 8:
            dilations = [1, 1, 2, 4]
            strides = [1, 2, 1, 1]
        else:
            raise Warning("output_stride must be 8 or 16!")
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3],
                                       bn_momentum=bn_momentum)

        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, bn_momentum=bn_momentum))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if 'resnet' in backbone or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# Encoder(CNN[ResNet]+ASPP)+Decoder
class DeepLabV3p(nn.Module):
    def __init__(self, base_model, in_channels, n_class, pretrained, out_dim=64, output_stride=8, projection_head=True):
        super(DeepLabV3p, self).__init__()
        self.resnet_dict = {
            "resnet50": ResNet(Bottleneck, [3, 4, 6, 3], 3,  output_stride),
            "resnet101": ResNet(Bottleneck, [3, 4, 23, 3], 3, output_stride)
        }
        if in_channels > 3:
            self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1, padding=0, stride=1)
        self.backbone = self._get_basemodel(base_model, pretrained)
        self.aspp = ASPP(base_model, output_stride=output_stride, BatchNorm=nn.BatchNorm2d)
        self.decoder = Decoder(base_model, nn.BatchNorm2d)
        # classsifier head
        self.seghead = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, n_class, kernel_size=1, stride=1))

        if projection_head:
            # projection head
            self.projhead = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm2d(256),
                nn.Conv2d(in_channels=256, out_channels=out_dim, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.projhead=None
        # for param in self.projhead.parameters():
        #     if isinstance(param, nn.Conv2d):
        #         torch.nn.init.kaiming_normal_(param.weight)
        #     elif isinstance(param, nn.BatchNorm2d):
        #         param.weight.data.fill_(1)
        #         param.bias.data.zero_()
            # param.requires_grad = False  # NOT update by gradient
        self._init_weight()
    def _get_basemodel(self, model_name, pretrained):
        try:
            backbone = self.resnet_dict[model_name]
            if pretrained:
                backbone = load_pretrained_model(backbone, model_name)
        except KeyError:
            raise NotImplementedError
        else:
            return backbone

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        if inputs.size()[1]>3:
            inputs = self.conv0(inputs)
        x, low_level_feat = self.backbone(inputs)
        x = self.aspp(x)
        if self.projhead is not None:
            feat = self.projhead(x)
            feat = F.interpolate(feat, size=inputs.size()[2:], mode='bilinear', align_corners=True)
        else:
            feat = None

        x = self.decoder(x, low_level_feat)
        x = self.seghead(x)
        x = F.interpolate(x, size=inputs.size()[2:], mode='bilinear', align_corners=True)

        if feat is not None:
            return feat, x
        else:
            return x


if __name__ == '__main__':
    model = DeepLabV3p('resnet101', in_channels=3, n_class=6, output_stride=8, pretrained=True, projection_head=False)

    img = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        x = model(img)

    # print(f.size())
    print(x.size())
