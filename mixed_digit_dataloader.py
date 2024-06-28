 
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import random, os
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import copy


from fedlab.utils.dataset import FMNISTPartitioner, CIFAR10Partitioner, CIFAR100Partitioner
from fedlab.utils.functional import partition_report   
from utils.sampling import  trainset_sampling_mixed_digit,testset_sampling_mixed_digit

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):#idxs对应的是dataset的索引
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.targets = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, targets = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.targets = np.concatenate([self.targets,targets], axis=0)
                else:
                    self.images, self.targets = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.targets = self.targets[:data_len]
            else:
                self.images, self.targets = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.targets = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.targets = self.targets.astype(np.compat.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, target

def get_mixed_digit_datasets(percent = 1):
    #define transform
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # MNIST
    mnist_trainset  = DigitsDataset(data_path="data/mixed_digit_dataset/MNIST", channels=1, percent=percent, train=True,  transform=transform_mnist)
    mnist_testset   = DigitsDataset(data_path="data/mixed_digit_dataset/MNIST", channels=1, percent=0.1, train=False, transform=transform_mnist)
    # SVHN
    svhn_trainset  = DigitsDataset(data_path='data/mixed_digit_dataset/SVHN', channels=3, percent=percent,  train=True,  transform=transform_svhn)
    svhn_testset   = DigitsDataset(data_path='data/mixed_digit_dataset/SVHN', channels=3, percent=0.1,  train=False, transform=transform_svhn)
    # USPS
    usps_trainset  = DigitsDataset(data_path='data/mixed_digit_dataset/USPS', channels=1, percent=percent,  train=True,  transform=transform_usps)
    usps_testset   = DigitsDataset(data_path='data/mixed_digit_dataset/USPS', channels=1, percent=0.1,  train=False, transform=transform_usps)
    # Synth Digits
    synth_trainset = DigitsDataset(data_path='data/mixed_digit_dataset/SynthDigits/', channels=3, percent=percent,  train=True,  transform=transform_synth)
    synth_testset = DigitsDataset(data_path='data/mixed_digit_dataset/SynthDigits/', channels=3, percent=0.1,  train=False, transform=transform_synth)
    # MNIST-M
    mnistm_trainset = DigitsDataset(data_path='data/mixed_digit_dataset/MNIST_M/', channels=3, percent=percent,  train=True,  transform=transform_mnistm)
    mnistm_testset  = DigitsDataset(data_path='data/mixed_digit_dataset/MNIST_M/', channels=3, percent=0.1,  train=False, transform=transform_mnistm)

    trainsets = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
    testsets  = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]

    return trainsets, testsets

def get_mixed_digit_client_dataloaders(args, reture_index=False):
    #get dataset
    trainsets, testsets = get_mixed_digit_datasets()
    datasets_name = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
    datasets_client_index = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
    #get number of classes
    num_classes = max(list(trainsets[0].targets))+1
    col_names = [f"class{i}" for i in range(num_classes)]
    # print(col_names)
    hist_color = '#4169E1'
    # plt.rcParams['figure.facecolor'] = 'white'

    clients_indexset = [ i for i in range(args.K)]
    clientnumbers_per_dataset = int(args.K/len(datasets_name))
    np.random.seed(args.seed-1)
    for i in range(len(datasets_name)):
        datasets_client_index[datasets_name[i]] = list(np.random.choice(clients_indexset, clientnumbers_per_dataset, replace=False))
        clients_indexset = list(set(clients_indexset) - set(datasets_client_index[datasets_name[i]]))

    # client index for each dataset
    clients_dataset_index = {i:[] for i in range(args.K)}
    for i in range(args.K):
        for dataset_name in datasets_client_index.keys():
            if i in datasets_client_index[dataset_name]:
                clients_dataset_index[i] = datasets_name.index(dataset_name)

    Dtrs = {i:[] for i in range(args.K)}
    # perform partition
    labeldir_parts = []
    labeldir_part_dfs = []
    for dataset_name, trainset in zip(datasets_name, trainsets):
        dataset_client_num = len(datasets_client_index[dataset_name])
        if 'iid' in args.split:
            labeldir_part = FMNISTPartitioner(trainset.targets, 
                                                        num_clients=dataset_client_num,
                                                        partition="iid",
                                                        seed=1)
        elif '_2' in args.split:
            labeldir_part = FMNISTPartitioner(trainset.targets,  
                                                    num_clients=dataset_client_num,
                                                    partition="noniid-#label", 
                                                    major_classes_num=2,
                                                    seed=3)
        elif 'dir' in args.split:
            labeldir_part = FMNISTPartitioner(trainset.targets, 
                                            num_clients=dataset_client_num,
                                            partition="noniid-labeldir", 
                                            dir_alpha=float(args.split[-3:]),
                                            seed=3)
        else:
            raise ValueError("Wrong split argument")
        # # generate partition report
        # csv_file = "data/fmnist/fmnist_noniid_labeldir_clients_10.csv"
        # partition_report(trainset.targets, labeldir_part.client_dict, 
        #                 class_num=num_classes, 
        #                 verbose=False, file=csv_file)

        # labeldir_part_df = pd.read_csv(csv_file,header=1)
        # labeldir_part_df = labeldir_part_df.set_index('client')
        # for col in col_names:
        #     labeldir_part_df[col] = (labeldir_part_df[col] * labeldir_part_df['Amount']).astype(int)
            
        # labeldir_parts.append(labeldir_part)
        # labeldir_part_dfs.append(labeldir_part_df)
        for i, k in enumerate(datasets_client_index[dataset_name]):

            if len(labeldir_part.client_dict[i]) % args.B == 1:
                labeldir_part.client_dict[i] = labeldir_part.client_dict[i][:-1]
            Dtrs[k] = DataLoader(DatasetSplit(trainsets[clients_dataset_index[k]], labeldir_part.client_dict[i]), batch_size=args.B, shuffle=True)
 
    # # client dataset sampling
    # trainset_sample_rate = args.trainset_sample_rate
    # rare_class_nums = 0
    # dict_users_train = {i: [] for i in range(args.K)}
    # for index, trainset in enumerate(trainsets):
    #     dict_users_train_part = trainset_sampling_mixed_digit(args, datasets_client_index[datasets_name[index]], trainset, trainset_sample_rate, rare_class_nums, labeldir_parts[index])
    #     for key in dict_users_train_part.keys():
    #         dict_users_train[key] = dict_users_train_part[key]
            
    # dict_test = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
    # for index, testset in enumerate(testsets):
    #         dict_test[datasets_name[index]] = testset_sampling_mixed_digit(args, testset, 100)
    if reture_index:
        return Dtrs, datasets_client_index, clients_dataset_index
    else:
        return Dtrs
