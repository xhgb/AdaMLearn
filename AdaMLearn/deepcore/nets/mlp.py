import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder
from backpack import backpack, extend

# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


''' MLP '''


class MLP(nn.Module):
    def __init__(self, channel, num_classes, im_size, hidden_channels=128, backpack: bool = False, record_embedding: bool = False,
                 no_grad: bool = False, pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("torchvison pretrained models not available.")
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(im_size[0] * im_size[1] * channel, hidden_channels, bias=False)
        self.fc_2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        if backpack:
            self.classifier = extend(nn.Linear(hidden_channels, num_classes, bias=False))
        else:
            self.classifier = nn.Linear(hidden_channels, num_classes, bias=False)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.classifier

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            out = x.view(x.size(0), -1)
            out = F.relu(self.fc_1(out))
            out = F.relu(self.fc_2(out))
            out = self.embedding_recorder(out)
            out = self.classifier(out)
        return out
