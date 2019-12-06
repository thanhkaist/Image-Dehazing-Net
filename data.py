import os
import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image


def get_train_dataloader(name, batch_size):
    if name == 'Indoor':
        dataset = ImageFolder('data/IndoorTrain', multi_scale=False, need_patch=True, augment=True)
    elif name == 'Outdoor':
        dataset = ImageFolder('data/OutdoorTrain', need_patch=True, augment=True)
    elif name == 'InOutdoor':
        dataset = ImageFolder('data/OutdoorTrain', need_patch=True, augment=True)
    else:
        raise Exception('Dataset is not supported')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             drop_last=True, shuffle=True, num_workers=8,
                                             pin_memory=True)
    return dataloader


def get_test_dataloader(name, batch_size=1):
    if name == 'Indoor':
        dataset = ImageFolder('data/IndoorTrain', multi_scale=False, need_patch=False, augment=False)
    elif name == 'Outdoor':
        dataset = ImageFolder('data/OutdoorTrain', multi_scale=False, need_patch=False, augment=False)
    else:
        raise Exception('Dataset is not supported')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             drop_last=True, shuffle=False, num_workers=8,
                                             pin_memory=False)
    return dataloader


class ImageFolder(data.DataLoader):

    def __init__(self, name='IndoorTrain', multi_scale=False, need_patch=False, augment=False,
                 transform=transforms.Compose([transforms.ToTensor()])):
        self.dirHazy = name + 'Hazy/'
        self.dirGT = name + 'GT/'
        self.transform = transform
        self.crop = need_patch
        self.crop_size = 512
        self.rotation = augment
        self.multiscale = multi_scale
        self.fileList = os.listdir(self.dirHazy)
        self.nTrain = len(self.fileList)

    def __getitem__(self, idx):
        hazy, gt = self.getFileName(idx)
        hazy = Image.open(hazy).convert('RGB')
        gt = Image.open(gt).convert('RGB')

        if self.rotation:
            degree = random.choice([0, 90, 180, 270])
            hazy = transforms.functional.rotate(hazy, degree)
            gt = transforms.functional.rotate(gt, degree)

        if self.transform:
            hazy = self.transform(hazy)
            gt = self.transform(gt)

        if self.crop:
            W = hazy.size()[1]
            H = hazy.size()[2]

            Ws = np.random.randint(0, W - self.crop_size - 1, 1)[0]
            Hs = np.random.randint(0, H - self.crop_size - 1, 1)[0]

            hazy = hazy[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
            gt = gt[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]

        if self.multiscale:
            H = gt.size()[1]
            W = gt.size()[2]
            hazy_s1 = transforms.ToPILImage()(hazy)
            gt_s1 = transforms.ToPILImage()(gt)
            hazy_s2 = transforms.ToTensor()(transforms.Resize([H // 2, W // 2])(hazy_s1)).mul(1.0)
            gt_s2 = transforms.ToTensor()(transforms.Resize([H // 2, W // 2])(gt_s1)).mul(1.0)
            hazy_s1 = transforms.ToTensor()(hazy_s1).mul(1.0)
            gt_s1 = transforms.ToTensor()(gt_s1).mul(1.0)

            return {'hazy_s1': hazy_s1, 'hazy_s2': hazy_s2,
                    'gt_s1': gt_s1, 'gt_s2': gt_s2, }, self.fileList[idx]
        else:
            return hazy, gt,self.fileList[idx]

    def __len__(self):
        return self.nTrain

    def getFileName(self, idx):
        name = self.fileList[idx]
        hazy_image = os.path.join(self.dirHazy, name)
        gt_image = os.path.join(self.dirGT, name)
        return hazy_image, gt_image
