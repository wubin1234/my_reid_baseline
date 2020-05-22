"""
@author: wubin
@connection: 799639771@qq.com
"""
import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck


def weight_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') !=-1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_channels = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == "resnet50":
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained ImageNet model......")
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_channels, self.num_classes)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_channels)
            self.bottleneck.bias.requires_grad_(False)
            self.classifier = nn.Linear(self.in_channels, self.num_classes, bias=False)
            # 手动初始化权重
            self.bottleneck.apply(weight_init_kaiming)
            self.bottleneck.apply(weight_init_classifier)

    def forward(self, x):
        global_feat = self.gap(self.base(x))    # (b, 2048,1,1)

        global_feat = global_feat.view(global_feat.shape[0], -1)   # flatten to (bs, 2048)
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            # 测试在BN层后的特征和BN层前的特征
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat


    def load_param(self, trained_path):
        para_dict = torch.load(trained_path)
        for i in para_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(para_dict[i])