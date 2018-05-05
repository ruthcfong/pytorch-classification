'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import os


__all__ = ['alexnet']

CHECKPOINT_DIR = '/home/ruthfong/pytorch-classification/pretrained'
model_name = 'alexnet'
model_urls = {model_name: {'cifar10': os.path.join(CHECKPOINT_DIR, 'cifar10', 
                                        '%s.pth.tar' % model_name),
                          'cifar100': os.path.join(CHECKPOINT_DIR, 'cifar100', 
                                        '%s.pth.tar' % model_name)
                          }
             }



class AlexNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=5),
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
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, dataset='cifar10', **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    if pretrained:
        if dataset == 'cifar10':
            model.features = nn.DataParallel(model.features)
        else:
            model = nn.DataParallel(model)
        checkpoint = torch.load(model_urls[model_name][dataset], 
                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if dataset == 'cifar10':
            model.features = model.features.module
        else:
            model = model.module
    return model.cpu()
