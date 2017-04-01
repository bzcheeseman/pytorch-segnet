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
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # first group

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # second group

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # third group

        self.encoder_4 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # fourth group

        self.unpool_1 = nn.MaxUnpool2d(2, stride=2)  # get masks
        self.unpool_2 = nn.MaxUnpool2d(2, stride=2)
        self.unpool_3 = nn.MaxUnpool2d(2, stride=2)
        self.unpool_4 = nn.MaxUnpool2d(2, stride=2)

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64)
        )  # first group

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64)
        )  # second group

        self.decoder_3 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64)
        )  # third group

        self.decoder_4 = nn.Sequential(
            nn.Conv2d(64, 3, 7, padding=3),
            nn.BatchNorm2d(3)
        )  # fourth group

        self.conv_classifier = nn.Conv2d(64, 11, 1)

    def forward(self, x):
        size_1 = x.size()
        x, indices_1 = self.encoder_1(x)

        size_2 = x.size()
        x, indices_2 = self.encoder_2(x)

        size_3 = x.size()
        x, indices_3 = self.encoder_3(x)

        size_4 = x.size()
        x, indices_4 = self.encoder_4(x)

        x = self.unpool_1(x, indices_4, output_size=size_4)
        x = self.decoder_1(x)

        x = self.unpool_2(x, indices_3, output_size=size_3)
        x = self.decoder_2(x)

        x = self.unpool_3(x, indices_2, output_size=size_2)
        x = self.decoder_3(x)

        x = self.unpool_4(x, indices_1, output_size=size_1)
        x = self.decoder_4(x)

        x = Funct.log_softmax(x)  # going to use NLLLoss - figure this out

        return x


if __name__ == "__main__":
    from utils import SegNetData
    import torch.optim as optim
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    import torchvision.transforms as t

    train_dataset = DataLoader(
        SegNetData("../data/train", transform=t.Compose([t.ToTensor()]),
                   target_transform=t.Compose([t.ToTensor()])),
        batch_size=4, shuffle=True, num_workers=4)
    test_dataset = DataLoader(
        SegNetData("../data/test", transform=t.Compose([t.ToTensor()]),
                   target_transform=t.Compose([t.ToTensor()])),
        batch_size=4, shuffle=True, num_workers=4)

    model = SegNet()
    model.train()

    criterion = nn.NLLLoss2d()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    max_epochs = 10
    for i in range(max_epochs):

        for j, data in enumerate(train_dataset, 0):
            running_loss = 0.0

            inputs, labels = data
            inputs = Variable(inputs)
            labels = Variable(labels)

            model.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i % 5000 == 4999:
                print('[epoch: %d, i: %5d] average loss: %.3f' % (i + 1, j + 1, running_loss / 5000))

                if running_loss / 5000 <= 0.005:
                    break

                running_loss = 0.0

    total_loss = 0.0
    for i, data in enumerate(test_dataset, 0):
        inputs, labels = data
        inputs.volatile = True
        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = model(inputs)

        total_loss += len(data) * criterion(outputs, labels).data

    print("Total Loss: {}".format(total_loss / len(test_dataset)))






