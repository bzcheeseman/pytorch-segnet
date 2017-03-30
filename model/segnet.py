#
# Created by Aman LaChapelle on 3/28/17.
#
# pytorch-segnet
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-segnet/LICENSE.txt
#

import torch
import torch.nn as nn
import torch.nn.functional as Funct

from collections import OrderedDict


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.Relu(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # first group

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.Relu(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # second group

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.Relu(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # third group

        self.encoder_4 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.Relu(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # fourth group

        self.decoder_1 = nn.Sequential(
            nn.MaxUnpool2d(2, stride=2),  # get mask from fourth group of encoder(?)
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm1d(64)
        )  # first group

        self.decoder_2 = nn.Sequential(
            nn.MaxUnpool2d(2, stride=2),  # get mask from fourth group of encoder
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm1d(64)
        )  # second group

        self.decoder_3 = nn.Sequential(
            nn.MaxUnpool2d(2, stride=2),  # get mask from fourth group of encoder
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm1d(64)
        )  # third group

        self.decoder_4 = nn.Sequential(
            nn.MaxUnpool2d(2, stride=2),  # get mask from fourth group of encoder
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm1d(64)
        )  # fourth group

        self.conv_classifier = nn.Conv2d(64, 11, 1)

        # pass through softmax

    def forward(self, x):
        x, indices_1 = self.encoder_1(x)
        x, indices_2 = self.encoder_2(x)
        x, indices_3 = self.encoder_3(x)
        x, indices_4 = self.encoder_4(x)

        x = self.decoder_1(x, indices_4)
        x = self.decoder_2(x, indices_3)
        x = self.decoder_3(x, indices_2)
        x = self.decoder_4(x, indices_1)
        x = Funct.log_softmax(x)  # going to use NLLLoss

        return x


if __name__ == "__main__":
    import torchvision  # https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/segnet_basic_train.prototxt




