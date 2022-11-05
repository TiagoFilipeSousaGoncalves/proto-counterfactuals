# PyTorch Imports
import torch.nn as nn
import torch.utils.model_zoo as model_zoo



# Dictionary w/ Model URLs
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

# model_dir = './pretrained_models'



# Dictionary w/ VGG configurations
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



# Class: VGG-features
class VGG_features(nn.Module):

    def __init__(self, cfg, cfg_type=None, batch_norm=False, init_weights=True, final_maxpool=False, final_relu=True):
        super(VGG_features, self).__init__()
        self.batch_norm = batch_norm
        self.kernel_sizes = []
        self.strides = []
        self.paddings = []
        self.features = self._make_layers(cfg, batch_norm, cfg_type, final_maxpool, final_relu)


        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def _make_layers(self, cfg, batch_norm, cfg_type=None, final_maxpool=False, final_relu=False):
        self.n_layers = 0
        layers = []
        in_channels = 3
        for i, v in enumerate(cfg):
            if v == 'M':
                # Removes final max pooling layer
                if i == len(cfg)-1 and not final_maxpool:
                    print("No final max pool")
                    continue
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

                self.kernel_sizes.append(2)
                self.strides.append(2)
                self.paddings.append(0)

            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    if i >= len(cfg)-2 and not final_relu:
                        print("No final relu")
                        layers += [conv2d]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]

                self.n_layers += 1

                self.kernel_sizes.append(3)
                self.strides.append(1)
                self.paddings.append(1)

                in_channels = v

        return nn.Sequential(*layers)


    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings


    def num_layers(self):
        '''
        the number of conv layers in the network
        '''
        return self.n_layers


    def __repr__(self):
        template = 'VGG{}, batch_norm={}'
        return template.format(self.num_layers() + 3, self.batch_norm)



# Class: VGG-vanilla
class VGG_vanilla(nn.Module):
    def __init__(self):
        super(VGG_vanilla, self).__init__()

        self.vgg19_f = vgg19_features(pretrained=True, include_classifier=True, final_maxpool=True, final_relu=True)

        # Taking linear layer sizes from VGG 19
        # self.addons = nn.Sequential(nn.Linear(512*7*7, 4096), nn.Linear(4096, 200))
        self.addons = nn.Linear(512*7*7, 200)


    def forward(self, x):
        x = self.vgg19_f(x)
        (_, C, H, W) = x.shape
        x = x.view(x.size(0), -1)
        x = self.addons(x)

        return x



# Function: VGG11-features
def vgg11_features(pretrained=False, **kwargs):
    # VGG 11-layer model (configuration "A")

    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['A'], batch_norm=False, **kwargs)

    if pretrained:
        # my_dict = model_zoo.load_url(model_urls['vgg11'], model_dir=model_dir)
        my_dict = model_zoo.load_url(model_urls['vgg11'])
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)

    return model



# Function: VGG11-BN-features
def vgg11_bn_features(pretrained=False, **kwargs):
    # VGG 11-layer model (configuration "A") with batch normalization

    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['A'], batch_norm=True, **kwargs)

    if pretrained:
        # my_dict = model_zoo.load_url(model_urls['vgg11_bn'], model_dir=model_dir)
        my_dict = model_zoo.load_url(model_urls['vgg11_bn'])
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    
    return model



# Function: VGG13-features
def vgg13_features(pretrained=False, **kwargs):
    # VGG 13-layer model (configuration "B")

    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['B'], batch_norm=False, **kwargs)

    if pretrained:
        # my_dict = model_zoo.load_url(model_urls['vgg13'], model_dir=model_dir)
        my_dict = model_zoo.load_url(model_urls['vgg13'])
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    
    return model



# Function: VGG13-BN-features
def vgg13_bn_features(pretrained=False, **kwargs):
    # VGG 13-layer model (configuration "B") with batch normalization

    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['B'], batch_norm=True, **kwargs)

    if pretrained:
        # my_dict = model_zoo.load_url(model_urls['vgg13_bn'], model_dir=model_dir)
        my_dict = model_zoo.load_url(model_urls['vgg13_bn'])
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)

    return model



# Function: VGG16-features
def vgg16_features(pretrained=False, final_maxpool=False, **kwargs):
    # VGG 16-layer model (configuration "D")

    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['D'], batch_norm=False, final_maxpool=final_maxpool, final_relu=True, **kwargs)

    if pretrained:
        # my_dict = model_zoo.load_url(model_urls['vgg16'], model_dir=model_dir)
        my_dict = model_zoo.load_url(model_urls['vgg16'])
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)

    return model



# Function: VGG16-BN-features
def vgg16_bn_features(pretrained=False, **kwargs):
    # VGG 16-layer model (configuration "D") with batch normalization

    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['D'], batch_norm=True, **kwargs)

    if pretrained:
        # my_dict = model_zoo.load_url(model_urls['vgg16_bn'], model_dir=model_dir)
        my_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    
    return model



# Function: VGG19-features
def vgg19_features(pretrained=False, include_classifier=False, final_maxpool=False, final_relu=True, **kwargs):
    # VGG 19-layer model (configuration "E")

    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['E'], batch_norm=False, cfg_type='E', final_maxpool=final_maxpool, final_relu=final_relu, **kwargs)

    if pretrained:
        # my_dict = model_zoo.load_url(model_urls['vgg19'], model_dir=model_dir)
        my_dict = model_zoo.load_url(model_urls['vgg19'])
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier') and not include_classifier:
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)

    return model



# Function: VGG19-BN-features
def vgg19_bn_features(pretrained=False, **kwargs):
    # VGG 19-layer model (configuration 'E') with batch normalization

    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['E'], batch_norm=True, **kwargs)

    if pretrained:
        # my_dict = model_zoo.load_url(model_urls['vgg19_bn'], model_dir=model_dir)
        my_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)

    return model
