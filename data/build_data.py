# UPDATED ON AUG 2ND
# @ WANGSHIROU
import os

from osgeo import gdal
import torch
from torch.utils import data
from torch.utils.data import Dataset, sampler
import numpy as np
import torchvision.transforms as transforms
import random


# --------------------------------------------------------------------------------
# Define transforms
# --------------------------------------------------------------------------------
class RandomFlipAndRotate(object):
    """
    Does data augmentation during training by randomly flipping (horizontal and vertical)
    and randomly rotating (0, 90, 180, 270 degrees).
    Keep in mind that pytorch samples are CxWxH.
    """
    def __call__(self, image):
        # randomly flip
        if np.random.rand() < 0.5:
            image = torch.flip(image, dims=[2])  # left-right
        if np.random.rand() < 0.5:
            image = torch.flip(image, dims=[1])  # up-down

        # randomly rotate
        rotations = random.choice([0, 1, 2, 3])
        if rotations > 0:
            image = torch.rot90(image, k=rotations, dims=(1, 2))

        return image

class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        return img


# --------------------------------------------------------------------------------
# Define batch transforms
# --------------------------------------------------------------------------------
def single_transform(image, label, logits):
    """
    Define single transform for image/label/logits
    """

    '''Geometric transform for image/label/logits'''
    # randomly flip
    if torch.rand(1) < 0.5:
        image = torch.flip(image, dims=[2])  # left-right
        label = torch.flip(label, dims=[2])
        logits = torch.flip(logits, dims=[2])

    if torch.rand(1) < 0.5:
        image = torch.flip(image, dims=[1])  # up-down
        label = torch.flip(label, dims=[1])
        logits = torch.flip(logits, dims=[1])

    # randomly rotate
    rotations = int(torch.rand(1) * 4)
    if rotations > 0:
        image = torch.rot90(image, k=rotations, dims=(1, 2))
        label = torch.rot90(label, k=rotations, dims=(1, 2))
        logits = torch.rot90(logits, k=rotations, dims=(1, 2))

    '''Color transform for image only'''
    # randomly Gaussian blur
    Gaussian_blur = transforms.GaussianBlur(kernel_size=(9,13), sigma=random.uniform(0.15, 1.15))
    image = Gaussian_blur(image)

    # randomly Color Jitter
    color_transform = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)  # For PyTorch 1.9/TorchVision 0.10 users
    image[:3] = color_transform(image[:3])

    return image, label, logits

def batch_transform(data, labels, logits):
    """
    Define batch transform for data/labels/logits
    """
    data_list, label_list, logits_list = [], [], []
    device = data.device

    for k in range(data.shape[0]):
        aug_data, aug_label, aug_logits = single_transform(data[k], labels[k].unsqueeze(0), logits[k].unsqueeze(0))
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    data_trans, label_trans, logits_trans = \
        torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
    return data_trans, label_trans, logits_trans

def batch_transform_imgonly(input):
    """
    Define batch transform for images only
    """

    data_list = []
    for k in range(input.shape[0]):
        image = input[k]
        # randomly flip
        if torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[2])  # left-right

        if torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[1])  # up-down

        # randomly rotate
        rotations = int(torch.rand(1)*4)
        if rotations > 0:
            image = torch.rot90(image, k=rotations, dims=(1, 2))

        # randomly Gaussian blur
        Gaussian_blur = transforms.GaussianBlur(kernel_size=(9,13), sigma=random.uniform(0.15, 1.15))
        image = Gaussian_blur(image)

        # randomly Color Jitter
        color_transform = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)  # For PyTorch 1.9/TorchVision 0.10 users
        image[:3] = color_transform(image[:3])

        data_list.append(image.unsqueeze(0))

    return torch.cat(data_list)


# --------------------------------------------------------------------------------
# Define datasets & dataloaders
# --------------------------------------------------------------------------------
class BuildDataset(Dataset):
    """
    Define dataset for training.
    """

    def __init__(self, root_folder, list_txt, baugment=True, norm=None):

        # Train set folder (images & labels)
        self.img_dir = root_folder + '/imgs'
        self.lab_dir = root_folder + '/labs'
        self.dsm_dir = root_folder + '/dsms'
        if not os.path.exists(self.dsm_dir):
            self.dsm_dir = None

        # Sample List
        samples = []
        with open(list_txt, 'r') as f:
            for file in f:
                samples.append(file[:-1])
        self.samples = samples

        # Define Transforms
        transform_list = []
        transform_list.append(ToFloatTensor())              # convert to tensor (C,B,W,H)
        if baugment:
            transform_list.append(RandomFlipAndRotate())    # random transform(weak): flip(H/V), rotate(0/90/180/270)

        transform = transforms.Compose(transform_list)
        self.transform = transform
        self.norm = norm

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):
        # print(idx)
        img = self.read_as_array(os.path.join(self.img_dir, self.samples[idx]))
        lab = self.read_as_array(os.path.join(self.lab_dir, self.samples[idx]))
        lab = np.expand_dims(lab, axis=0)

        if self.dsm_dir is None:
            tile = np.concatenate([img, lab], axis=0)  # Make sure same transforms
        else:
            dsm = self.read_as_array(os.path.join(self.dsm_dir, self.samples[idx]))
            dsm = np.expand_dims(dsm, axis=0)
            tile = np.concatenate([img, dsm, lab], axis=0)  # Make sure same transforms
        tile = self.transform(tile)

        img = tile[:-1, :, :]
        lab = tile[-1, :, :]
        if self.norm is not None:
            img = self.data_normal(img, self.norm)

        return img, lab

    def data_normal(self, feature_files, norm):
        for i in range(feature_files.shape[0]):
            tile = feature_files[i]

            d_min = norm[i*2]
            d_max = norm[i*2+1]

            dst = d_max - d_min
            if dst == 0:
                feature_files[i] = tile
                continue

            feature_files[i] = (tile - d_min) / dst

        return feature_files

    def read_as_array(self, filename):
        """
            input: filename
            return: data array
        """
        img_tif = gdal.Open(filename)
        if img_tif is None:
            print(filename + " Open Failure!")
        width = img_tif.RasterXSize
        height = img_tif.RasterYSize
        arrdata = img_tif.ReadAsArray(0, 0, width, height)

        return arrdata

def build_semi_dataloader(args):
    """
    Get Dataloader for semi-supervised setting.
    """

    # lab/unlab sets
    train_l_dataset = BuildDataset(args.train_data, args.lab_set, baugment=True, norm=args.norm)
    train_u_dataset = BuildDataset(args.train_data, args.unlab_set, baugment=True, norm=args.norm)

    # Define Dataloaders
    if len(train_l_dataset) < args.batch_size:
        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=args.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=True,
                                          num_samples=args.batch_size),drop_last=True)
    else:
        train_l_loader = data.DataLoader(
            train_l_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    train_u_loader = data.DataLoader(
        train_u_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    return train_l_loader, train_u_loader

def build_supervised_dataloader(args):
    """
    Get Dataloader for supervised setting.
    """

    # train set
    train_dataset = BuildDataset(args.train_data, args.train_set, baugment=True, norm=args.norm)

    # Define Dataloaders
    if len(train_dataset) < args.batch_size:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler.RandomSampler(data_source=train_dataset,
                                          replacement=True,
                                          num_samples=args.batch_size),drop_last=True)
    else:
        train_loader = data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    return train_loader


class TestDataset(Dataset):
    """
    Define dataset for train/test.
    """

    def __init__(self, root_folder, list_txt, norm=None, lab_erode=False):

        # Train set folder (images & labels)
        self.img_dir = root_folder + '/imgs'
        if lab_erode:
            self.lab_dir = root_folder + '/lab_erode'
        else:
            self.lab_dir = root_folder + '/labs'
        self.dsm_dir = root_folder + '/dsms'
        if not os.path.exists(self.dsm_dir):
            self.dsm_dir=None

        # Sample List
        samples = []
        with open(list_txt, 'r') as f:
            for file in f:
                samples.append(file[:-1])
        self.samples = samples

        # Define Transforms
        transform_list = []
        transform_list.append(ToFloatTensor())              # convert to tensor (C,B,W,H)
        transform = transforms.Compose(transform_list)
        self.transform = transform
        self.norm = norm

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        lab = self.read_as_array(os.path.join(self.lab_dir, self.samples[idx]))
        img_tif = gdal.Open(os.path.join(self.img_dir, self.samples[idx]))

        width = img_tif.RasterXSize
        height = img_tif.RasterYSize
        img = img_tif.ReadAsArray(0, 0, width, height)
        proj = img_tif.GetProjection()
        geotrans = list(img_tif.GetGeoTransform())

        lab = np.expand_dims(lab, axis=0)
        if self.dsm_dir is not None:
            dsm = self.read_as_array(os.path.join(self.dsm_dir, self.samples[idx]))
            dsm = np.expand_dims(dsm, axis=0)
            tile = np.concatenate([img, dsm, lab], axis=0)  # Make sure same transforms\
        else:
            tile = np.concatenate([img, lab], axis=0)  # Make sure same transforms\
        tile = self.transform(tile)

        img = tile[:-1, :, :]
        lab = tile[-1, :, :]

        if self.norm is not None:
            img = self.data_normal(img, self.norm)

        return img, lab, self.samples[idx], proj, geotrans


    def data_normal(self, feature_files, norm):
        for i in range(feature_files.shape[0]):
            tile = feature_files[i]

            d_min = norm[i*2]
            d_max = norm[i*2+1]

            dst = d_max - d_min
            if dst == 0:
                feature_files[i] = tile
                continue

            feature_files[i] = (tile - d_min) / dst

        return feature_files

    def read_as_array(self, filename):
        """
            input: filename
            return: data array
        """
        img_tif = gdal.Open(filename)
        if img_tif is None:
            print(filename + " Open Failure!")
        width = img_tif.RasterXSize
        height = img_tif.RasterYSize
        arrdata = img_tif.ReadAsArray(0, 0, width, height)

        return arrdata


def build_test_dataloader(test_data, test_set, batch_size, norm, lab_erode):
    test_dataset = TestDataset(test_data, test_set, norm=norm, lab_erode=lab_erode)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return test_loader
