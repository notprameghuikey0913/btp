import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
affine_par = True
import functools

from net.cc_attention import CrissCrossAttention
from utils.pyt_utils import load_model
import net.xception as xception

from inplace_abn import InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
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

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

class SkinCCNet(nn.Module):
    def __init__(self, num_classes, recurrence):
        self.inplanes = 128
        super(SkinCCNet, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        self.aspp = ASPP(dim_in=2048, dim_out=256, rate=16//16, bn_mom = 0.99)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=16//4)
        self.head = RCCAModule(2048, 512, num_classes)

        self.shortcut_conv = nn.Sequential(nn.Conv2d(256, 48, 1, 1, padding=1//2, bias=True),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),
		)

        self.backbone = xception.Xception(os=16)
        self.backbone_layers = self.backbone.get_layers()
        self.recurrence = recurrence


    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp,feature_shallow],1)
        
        x = self.head(feature_cat, self.recurrence)

        return x


class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch3 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch4 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out)
		self.branch5_relu = nn.ReLU(inplace=True)
		self.conv_cat = nn.Sequential(
			nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)


	def forward(self, x):
		[b, c, row, col] = x.size()
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
		global_feature = torch.mean(x, 2, True)
		global_feature = torch.mean(global_feature, 3, True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class SkinCCNet_en(nn.Module):
    def __init__(self, num_classes, recurrence):
        self.inplanes = 128
        super(SkinCCNet_en, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        self.aspp = ASPP(dim_in=2048, dim_out=256, rate=16//16, bn_mom = 0.99)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=16//4)
        self.head = RCCAModule(2048, 512, num_classes)

        self.cam_conv = nn.Sequential(nn.Conv2d(256+1, 256, 1, 1, padding=1//2, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
		)

        self.shortcut_conv = nn.Sequential(nn.Conv2d(256, 48, 1, 1, padding=1//2, bias=True),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),
		)

        self.backbone = xception.Xception(os=16)
        self.backbone_layers = self.backbone.get_layers()
        self.recurrence = recurrence


    def forward(self, x, cla_cam):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)

        feature_cat0 = torch.cat([feature_aspp, cla_cam], 1)
        feature_cam = self.cam_conv(feature_cat0)
        feature_cam = self.upsample_sub(feature_cam)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp,feature_shallow],1)
        
        x = self.head(feature_cat, self.recurrence)

        return x