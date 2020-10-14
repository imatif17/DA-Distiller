import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.utils.data as data
import torch
from PIL import Image
import os
import numpy as np


def get_source_target_loader(dataset_name, source_path, target_path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):

    source_dataloader = None
    if dataset_name == "Office31":
        if source_path != "":
            source_dataloader = office_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = office_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = office_test_loader(target_path, batch_size, num_workers, pin_memory)

    elif dataset_name == "ImageClef":
        if source_path != "":
            source_dataloader = imageclef_train_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = imageclef_train_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = imageclef_test_loader(target_path, batch_size, num_workers, pin_memory)
    else:
        raise("Dataset not handled")

    return source_dataloader, target_dataloader, target_testloader

def get_m_source_target_loader(dataset_name, source_path_1, source_path_2, target_path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    print(batch_size)
    source_dataloader_1 = None
    source_dataloader_2 = None
    if dataset_name == "Office31":
        if source_path_1 != "":
            source_dataloader_1 = office_loader(source_path_1, batch_size, num_workers, pin_memory, drop_last)
        if source_path_2 != "":
            source_dataloader_2 = office_loader(source_path_2, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = office_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = office_test_loader(target_path, batch_size, num_workers, pin_memory)

    elif dataset_name == "ImageClef":
        if source_path_1 != "":
            source_dataloader_1 = imageclef_train_loader(source_path_1, batch_size, num_workers, pin_memory, drop_last)
        if source_path_2 != "":
            source_dataloader_2 = imageclef_train_loader(source_path_2, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = imageclef_train_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = imageclef_test_loader(target_path, batch_size, num_workers, pin_memory)
    else:
        raise("Dataset not handled")

    return source_dataloader_1, source_dataloader_2, target_dataloader, target_testloader

def get_source_m_target_loader(dataset_name, source_path, target_path_s, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):

    targets_dataloader = [0] * len(target_path_s)
    targets_testloader = [0] * len(target_path_s)

    source_dataloader = None
    if dataset_name == "Office31":
        if source_path != "":
            source_dataloader = office_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        for i, p in enumerate(target_path_s):
            targets_dataloader[i] = office_loader(p, batch_size, num_workers, pin_memory, drop_last)
            targets_testloader[i] = office_test_loader(p, batch_size, num_workers, pin_memory)


    elif dataset_name == "ImageClef":
        if source_path != "":
            source_dataloader = imageclef_train_loader(source_path, batch_size, num_workers, pin_memory, drop_last)

        for i, p in enumerate(target_path_s):
            targets_dataloader[i] = imageclef_train_loader(p, batch_size, num_workers, pin_memory, drop_last)
            targets_testloader[i] = imageclef_test_loader(p, batch_size, num_workers, pin_memory)

    else:
        raise("Dataset not handled")

    return source_dataloader, targets_dataloader, targets_testloader

def get_source_concat_target_loader(dataset_name, source_path, target_path_1, target_path_2, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):

    source_dataloader = None
    if dataset_name == "Office31":
        if source_path != "":
            source_dataloader = office_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = office_m_concat_loader(target_path_1, target_path_2, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = office_m_concat_test_loader(target_path_1, target_path_2, batch_size, num_workers, pin_memory)

    elif dataset_name == "ImageClef":
        if source_path != "":
            source_dataloader = imageclef_train_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        raise ("To be implemented")
    else:
        raise("Dataset not handled")

    return source_dataloader, target_dataloader, target_testloader

def get_train_test_loader(dataset_name, data_path, batch_size=16, num_workers=1, pin_memory=False):
    if dataset_name == "Office31":
        train_loader, test_loader = office_train_test_loader(data_path, batch_size, num_workers, pin_memory)


    elif dataset_name == "ImageClef":
        train_loader, test_loader = imageclef_train_test_loader(data_path, batch_size, num_workers, pin_memory)
    else:
        raise("Dataset not handled")

    return train_loader, test_loader

class MnistMDataset(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data

def office_train_test_loader(path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(path,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    train_set, test_set = data.random_split(dataset,
                                            [int(0.7 * dataset.__len__()), dataset.__len__() - int(0.7 * dataset.__len__())])

    train_loader = data.DataLoader(train_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    test_loader = data.DataLoader(test_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    return train_loader, test_loader

def office_loader(path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(path,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    dataset.num_classes = len(dataset.classes)

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)


def office_m_concat_loader(path_1, path_2, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset_1 = datasets.ImageFolder(path_1,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    dataset_2 = datasets.ImageFolder(path_2,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
    dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2])
    dataset.num_classes = len(dataset_1.classes)

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def office_test_loader(path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(path,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

def office_m_concat_test_loader(path_1, path_2, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset_1 = datasets.ImageFolder(path_1,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    dataset_2 = datasets.ImageFolder(path_2,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2])

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

def imageclef_train_loader(path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = CLEFImage(path, transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]))

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def imageclef_test_loader(path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = CLEFImage(path, transforms.Compose([
                           transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           normalize,
                       ]))

    return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

def imageclef_train_test_loader(path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = CLEFImage(path,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    train_set, test_set = data.random_split(dataset,
                                            [int(0.7 * dataset.__len__()), dataset.__len__() - int(0.7 * dataset.__len__())])

    train_loader = data.DataLoader(train_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    test_loader = data.DataLoader(test_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    return train_loader, test_loader

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_imageclef_dataset(path):
    images = []
    dataset_name = path.split("/")[-1]
    label_path = os.path.join(path, ".." , "list", "{}List.txt".format(dataset_name))
    image_folder = os.path.join(path, "..", "{}".format(dataset_name))
    labeltxt = open(label_path)
    for line in labeltxt:
        pre_path, label = line.strip().split(' ')
        image_name = pre_path.split("/")[-1]
        image_path = os.path.join(image_folder, image_name)

        gt = int(label)
        item = (image_path, gt)
        images.append(item)
    return images

class CLEFImage(data.Dataset):
    def __init__(self, root, transform=None, image_loader=default_loader):
        imgs = make_imageclef_dataset(root)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.image_loader = image_loader
        self.num_classes = 12

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.image_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
