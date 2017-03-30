#
# Created by Aman LaChapelle on 3/30/17.
#
# pytorch-segnet
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-segnet/LICENSE.txt
#

# Partly copied from the torchvision.datasets.ImageFolder

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(d):
    images = []
    annot = []


    for root, _, fnames in sorted(os.walk(d)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)

    for root, _, fnames in sorted(os.walk("%sannot" % d)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                annot.append(item)

    return images, annot


def default_loader(path):
    return Image.open(path).convert('RGB')


class SegNetData(data.Dataset):

    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 loader=default_loader):

        imgs, annot = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.annot = annot
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path_to_img = self.imgs[index]
        path_to_target = self.annot[index]
        img = self.loader(path_to_img)
        target = self.loader(path_to_target)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

