# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project Imports
from model_resnet_utilities import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from model_densenet_utilities import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from model_vgg_utilities import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features, vgg19_features, vgg19_bn_features



# Class: DenseNet Model
class DenseNet(torch.nn.Module):
    def __init__(self, backbone, channels, height, width, nr_classes, pretrained=True):
        super(DenseNet, self).__init__()

        # Init variables
        self.backbone = backbone
        self.pretrained = pretrained
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes
        self.features = None


        # Init modules
        # Backbone to extract features
        assert self.backbone in ('densenet121', 'densenet161', 'densenet169', 'densenet201'), "Please provide a valid backbone name (i.e., 'densenet121', 'densenet161', 'densenet169', 'densenet201')"
        if self.backbone == 'densenet121':
            self.features = densenet121_features(pretrained=self.pretrained)
        elif self.backbone == 'densenet161':
            self.features = densenet161_features(pretrained=self.pretrained)
        elif self.backbone == 'densenet169':
            self.features = densenet169_features(pretrained=self.pretrained)
        else:
            self.features = densenet201_features(pretrained=self.pretrained)


        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.features(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create a linear layer for classification (we remove bias to make it comparable to the ProtoPNet)
        self.last_layer = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes, bias=False)

        return


    # Method: conv_features
    def conv_features(self, inputs):
        # Compute Backbone features
        conv_features = self.features(inputs)

        return conv_features


    # Method: forward
    def forward(self, inputs):
        # Compute Backbone features
        features = self.features(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.last_layer(features)

        return outputs



# Class: ResNet Model
class ResNet(torch.nn.Module):
    def __init__(self, backbone, channels, height, width, nr_classes, pretrained=True):
        super(ResNet, self).__init__()

        # Init variables
        self.backbone = backbone
        self.pretrained = pretrained
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes
        self.features = None


        # Init modules
        # Backbone to extract features
        assert self.backbone in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'), "Please provide a valid backbone name (i.e., 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')"
        if self.backbone == 'resnet18':
            self.features = resnet18_features(pretrained=self.pretrained)
        elif self.backbone == 'resnet34':
            self.features = resnet34_features(pretrained=self.pretrained)
        elif self.backbone == 'resnet50':
            self.features = resnet50_features(pretrained=self.pretrained)
        elif self.backbone == 'resnet101':
            self.backbone = resnet101_features(pretrained=self.pretrained)
        else: 
            self.backbone = resnet152_features(pretrained=self.pretrained)


        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.features(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create a linear layer for classification (we remove bias to make it comparable to the ProtoPNet)
        self.last_layer = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes, bias=False)

        return


    # Method: conv_features
    def conv_features(self, inputs):
        # Compute Backbone features
        conv_features = self.features(inputs)

        return conv_features


    # Method: forward
    def forward(self, inputs):
        # Compute Backbone features
        features = self.features(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.last_layer(features)

        return outputs



# Class: VGG Model
class VGG(torch.nn.Module):
    def __init__(self, backbone, channels, height, width, nr_classes, pretrained=True):
        super(VGG, self).__init__()

        # Init variables
        self.backbone = backbone
        self.pretrained = pretrained
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes
        self.features = None


        # Init modules
        # Backbone to extract features
        assert self.backbone in ('vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'), "Please provide a valid backbone name (i.e., 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn')"
        if self.backbone == 'vgg11':
            self.features = vgg11_features(pretrained=self.pretrained)
        elif self.backbone == 'vgg11_bn':
            self.features = vgg11_bn_features(pretrained=self.pretrained)
        elif self.backbone == 'vgg13':
            self.features = vgg13_features(pretrained=self.pretrained)
        elif self.backbone == 'vgg13_bn':
            self.features = vgg13_bn_features(pretrained=self.pretrained)
        elif self.backbone == 'vgg16':
            self.features = vgg16_features(pretrained=self.pretrained)
        elif self.backbone == 'vgg16_bn':
            self.features = vgg16_bn_features(pretrained=self.pretrained)
        elif self.backbone == 'vgg19':
            self.features = vgg19_features(pretrained=self.pretrained)
        else:
            self.features = vgg19_bn_features(pretrained=self.pretrained)


        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.features(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create a linear layer for classification (we remove bias to make it comparable to the ProtoPNet)
        self.last_layer = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes, bias=False)

        return


    # Method: conv_features
    def conv_features(self, inputs):
        # Compute Backbone features
        conv_features = self.features(inputs)

        return conv_features


    # Method: forward
    def forward(self, inputs):
        # Compute Backbone features
        features = self.features(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.last_layer(features)

        return outputs
