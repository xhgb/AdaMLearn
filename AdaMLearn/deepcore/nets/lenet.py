import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder
import torch
from backpack import backpack, extend


# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

class LeNet(nn.Module):
    def __init__(self, channel, num_classes, backpack=False, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("torchvison pretrained models not available.")
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 40, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, 0.001 / 9.0, 0.75, 1),
            nn.Dropout(0),
            nn.MaxPool2d(3, 2, padding=1),

            nn.Conv2d(40, 100, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, 0.001 / 9.0, 0.75, 1),
            nn.Dropout(0),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.features_lin = nn.Sequential(
            nn.Linear(6400, 1600, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0),

            nn.Linear(1600, 800, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0),
        )
        self.classifier = torch.nn.ModuleList()
        if backpack:
            for i in range(20):
                self.classifier.append(extend(nn.Linear(800, num_classes, bias=False)))
        else:
            for i in range(20):
                self.classifier.append(nn.Linear(800, num_classes, bias=False))

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self, task_id):
        return self.classifier[task_id]

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            y = []
            x = self.features(x)
            x = x.reshape(x.size(0), -1)
            x = self.features_lin(x)
            x = self.embedding_recorder(x)
            for i in range(20):
                y.append(self.classifier[i](x))
        return y


class LeNet_original(nn.Module):
    def __init__(self, channel, num_classes, backpack=False, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("torchvison pretrained models not available.")
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 20, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, 0.001 / 9.0, 0.75, 1),
            nn.Dropout(0),
            nn.MaxPool2d(3, 2, padding=1),

            nn.Conv2d(20, 50, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, 0.001 / 9.0, 0.75, 1),
            nn.Dropout(0),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.features_lin = nn.Sequential(
            nn.Linear(3200, 800, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0),

            nn.Linear(800, 500, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0),
        )
        self.classifier = torch.nn.ModuleList()
        if backpack:
            for i in range(20):
                self.classifier.append(extend(nn.Linear(500, num_classes, bias=False)))
        else:
            for i in range(20):
                self.classifier.append(nn.Linear(500, num_classes, bias=False))

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self, task_id):
        return self.classifier[task_id]

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            y = []
            x = self.features(x)
            x = x.reshape(x.size(0), -1)
            x = self.features_lin(x)
            x = self.embedding_recorder(x)
            for i in range(20):
                y.append(self.classifier[i](x))
        return y
