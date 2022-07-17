"""
   CIFAR-10 CIFAR-100, Tiny-ImageNet data loader
"""

import random
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
from PIL import Image

class Sampler(object):
    def __init__(self, data_source):
        pass
    def __len__(self):
        raise NotImplementedError

class Mysampler(torch.utils.data.Sampler):
    def __init__(self, data_source,split):
        self.data_source = data_source
        self.split = split
    def __iter__(self):
        return iter(range(len(self.data_source))[:self.split])
    def __len__(self):
        return len(self.data_source)

class Mysampler_v(torch.utils.data.Sampler):
    def __init__(self, data_source,split):
        self.data_source = data_source
        self.split = split
    def __iter__(self):
        return iter(range(len(self.data_source))[self.split:])
    def __len__(self):
        return len(self.data_source)

def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    # using random crops and horizontal flip for train set
    if params.augmentation:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # ************************************************************************************
    if params.dataset == 'cifar-10':
        trainset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=True,
                                                download=False, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=False,
                                              download=False, transform=dev_transformer)
    elif params.dataset == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
        trainset = torchvision.datasets.MNIST(root='./data/data-mnist', train=True,
                                                download=False, transform=trans_mnist)
        devset = torchvision.datasets.MNIST(root='./data/data-mnist', train=False,
                                              download=False, transform=trans_mnist)
    # ************************************************************************************
    elif params.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=False,
                                              download=True, transform=dev_transformer)
    elif params.dataset == 'cifar10':
        data_dir = '/home/yejingwen/Nasty-Teacher/data/data-cifar10/cifar-10-batches-all/'
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'test/'
        trainset = torchvision.datasets.ImageFolder(train_dir,train_transformer)
        devset = torchvision.datasets.ImageFolder(test_dir, dev_transformer)
    # ************************************************************************************
    elif params.dataset == 'tiny_imagenet':
        #data_dir = './data/tiny-imagenet-200/'
        data_dir = '/home/yejingwen/Nasty-Teacher/data/data-cifar10/cifar-10-batches-all/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'test/'
        trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])
        #print(devset.classes)
        #trainset = tiny_imagenet_data('train', train_dir, data_transforms['train'])
        #devset = tiny_imagenet_data('val', test_dir, data_transforms['val'])
    elif params.dataset == 'cifarsub3':
        data_dir = '/home/yejingwen/Nasty-Teacher/data/data-cifar10/cifarsubset3/'
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'test/'
        trainset = torchvision.datasets.ImageFolder(train_dir,train_transformer)
        devset = torchvision.datasets.ImageFolder(test_dir, dev_transformer)

    elif params.dataset == 'cifarsub7':
        data_dir = '/home/yejingwen/Nasty-Teacher/data/data-cifar10/cifarsubset7/'
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'test/'
        trainset = torchvision.datasets.ImageFolder(train_dir,train_transformer)
        devset = torchvision.datasets.ImageFolder(test_dir, dev_transformer)
    if params.dep_size > 0:
        trainsize = len(trainset)
        split = int(trainsize/10*params.dep_size)
        indices = range(trainsize)
        #train_sampler = Mysampler(indices,split)
        train_sampler = SubsetRandomSampler(indices[:split])

        devsize = len(devset)
        split = int(devsize/10*params.dep_size)
        indices = range(devsize)
        dev_sampler = Mysampler(indices,split)
        dev_sampler2 = Mysampler_v(indices,split)        
        for x in dev_sampler:
            print(x)
        #print(len(train_sampler))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
                                              shuffle=False, sampler = train_sampler,num_workers=params.num_workers)

    #trainloader = torch.utils.data.DataLoader(trainloader, batch_size=params.batch_size,
    #                                          shuffle=True,num_workers=params.num_workers)
    print(len(trainloader))
    #devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
    #                                        shuffle=False, num_workers=params.num_workers)

    if types == 'train':
        dl = trainloader
    elif types == 'dep':
        dl = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
                                            shuffle=False, sampler = dev_sampler, num_workers=params.num_workers)
    else:
        dl = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
                                            shuffle=False, sampler = dev_sampler2, num_workers=params.num_workers)

    return dl


def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    # using random crops and horizontal flip for train set
    if params.augmentation:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # ************************************************************************************
    if params.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=True,
                                                download=False, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=False,
                                              download=False, transform=dev_transformer)

    # ************************************************************************************
    elif params.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=True,
                                                download=False, transform=train_transformer)
        devset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=False,
                                              download=False, transform=dev_transformer)

    # ************************************************************************************
    elif params.dataset == 'tiny_imagenet':
        data_dir = './data/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'val/'
        trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
        print(trainset.classes)
        devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    print(indices)
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)
    print(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def fetch_subclass_dataloader(types, params):

    """
    Use only a subclass of dataset for training, depending on params.subset_percent
    """

    # using random crops and horizontal flip for train set
    if params.augmentation:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # ************************************************************************************
    if params.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=True,
                                                download=False, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=False,
                                              download=False, transform=dev_transformer)

    # ************************************************************************************
    elif params.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=True,
                                                download=False, transform=train_transformer)
        devset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=False,
                                              download=False, transform=dev_transformer)

    # ************************************************************************************
    elif params.dataset == 'tiny_imagenet':
        data_dir = './data/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'val/'
        trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])

    trainset_size = len(trainset)/2 #set 50% class for training the subclass network
    print(trainset_size)
    trainset_size = int(trainset_size)
    indices = list(range(trainset_size))
    #print(indices)
    #split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)
    #print(indices)

    #train_sampler = SubsetRandomSampler(indices[:split])
    train_sampler = SubsetRandomSampler(indices)
    print(trainset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def generate_noise(params):
    data_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    if params.dataset == 'cifar10' or params.dataset == 'cifar100':
        random.seed(1)
        noisy_img = np.random.randint(0,255,size=(32,32,3))
        return data_transformer(noisy_img/255.0).float()
    else:
        noisy_img = np.random.randint(0,255,size=(64,64,3))
        return data_transformer(noisy_img/255.0).float()

class tiny_imagenet_data(torch.utils.data.Dataset):
    def __init__(self, type, path, transform):
        self.type = type
        labels_t = []
        image_names = []
        with open(os.path.join(path, '..','wnids.txt')) as wnid:
            for line in wnid:
                labels_t.append(line.strip('\n'))
            
        if type == 'train':
            for label in labels_t:
                txt_path = os.path.join(path, label, label+'_boxes.txt')
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        image_name.append(line.strip('\n').split('\t')[0])
                image_names.append(image_name)
            i = 0
            self.images = []
            self.labels = []
            for label in labels_t:
                for image_name in image_names[i]:
                    image_path = os.path.join(path, label, 'images', image_name) 
                    self.images.append(Image.fromarray(cv2.imread(image_path)))
                    self.labels.append(i)
                i = i + 1
            self.labels = np.array(self.labels)
            
        elif type == 'val':
            self.labels_t = []
            self.labels = []
            val_names = []
            with open(os.path.join(path, 'val_annotations.txt')) as txt:
                for line in txt:
                    val_names.append(line.strip('\n').split('\t')[0])
                    self.labels_t.append(line.strip('\n').split('\t')[1])
            for i in range(len(self.labels_t)):
                for i_t in range(len(labels_t)):
                    if self.labels_t[i] == labels_t[i_t]:
                        self.labels.append(i_t)
            self.labels = np.array(self.labels)
            self.images = []
            for val_image in val_names:
                val_image_path = os.path.join(path, 'images', val_image)
                self.images.append(Image.fromarray(cv2.imread(val_image_path)))
        
        self.transform = transform
        
    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        return self.transform(image), label
        
    def __len__(self):
        return self.labels.shape[0]