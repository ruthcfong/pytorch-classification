'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import os
from alexnet import AlexNet, alexnet


__all__ = ['truncated_alexnet']


class TruncatedAlexNet(AlexNet):

    def __init__(self, num_classes=10):
        super(TruncatedAlexNet, self).__init__(num_classes=10)
        if module_name != 'classifier':
            module_name, module_index = module_name.split('.')
        assert(module_name in ['features', 'classifier'])

        features = [
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        if module_name == 'features':
            self.features = nn.Sequential(*features[:module_index+1])
            self.classifier is None
        else:
            self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        if self.classifier is None:
            return x
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def truncated_alexnet(module_name, pretrained=False, dataset='cifar10', num_classes=10):
    model = TruncatedAlexNet(module_name, num_classes=num_classes)
    if pretrained:
        pretrained_model = alexnet(pretrained=True, dataset=dataset, num_classes=10)
        state_dict = model.state_dict()
        pretrained_state_dict = pretrained_model.state_dict()
        for k in pretrained_state_dict.keys():
            if k in state_dict.keys() and state_dict[k].shape == pretrained_state_dict[k].shape:
                state_dict[k] = pretrained_state_dict[k]
        model.load_state_dict(state_dict)
    return model
