# Follows https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0213539&type=printable
# This is their "optimal net" from figure 1

import torch
from torch import nn
import torch.nn.functional as F

class DeepVess(nn.Module):
    def __init__(self,
                 in_channels=1,
                 n_classes=7,
                 padding=False):

        super(DeepVess, self).__init__()
        self.padding = padding
        self.n_classes = n_classes

        # First Set
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=32,
                               kernel_size=3,
                               padding=int(padding))
        self.conv2 = nn.Conv3d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding=int(padding))
        self.conv3 = nn.Conv3d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding=int(padding))

        # Second Set
        self.conv4 = nn.Conv3d(in_channels=32,
                               out_channels=64,
                               kernel_size=[1, 3, 3])
        self.conv5 = nn.Conv3d(in_channels=64,
                               out_channels=64,
                               kernel_size=[1, 3, 3])

        #self.before_last = nn.Linear(64*126*126, 1024)#, kernel_size=1)
        #self.last = nn.Linear(1024, n_classes*126*126)
        self.before_last = nn.Linear(1024, 1024)#, kernel_size=1)
        self.last = nn.Linear(1024, n_classes*5*5)

    def forward(self, x):
        #print("\tIn Model: input size", x.size())
        # First set
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool3d(x, kernel_size=[1, 2, 2])

        # Second Set
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool3d(x, kernel_size=[1, 2, 2])

        #x = x.view(-1, 64*126*126)
        x = x.view(-1, 1024)
        x = F.relu(self.before_last(x))
        x = self.last(x)
        x = x.view(-1, self.n_classes, 5, 5)
        return x

