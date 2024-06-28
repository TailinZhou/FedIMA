 
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


from fedlab.utils.dataset import FMNISTPartitioner, PACSPartitioner
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


def get_PACS_datasets(random_seed=1):
    data_base_path = './data'
    # means and standard deviations ImageNet because the network is pretrained
    means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # Define transforms to apply to each image
    transf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
                                transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                transforms.Normalize(means,stds) # Normalizes tensor with mean and standard deviation
    ])
    # Define datasets root
    DIR_PHOTO = data_base_path+'/PACS/photo'
    DIR_ART = data_base_path+'/PACS/art_painting'
    DIR_CARTOON = data_base_path+'/PACS/cartoon'
    DIR_SKETCH = data_base_path+'/PACS/sketch'

    # Prepare Pytorch train/test Datasets
    photo_dataset = torchvision.datasets.ImageFolder(DIR_PHOTO, transform=transf)
    art_dataset = torchvision.datasets.ImageFolder(DIR_ART, transform=transf)
    cartoon_dataset = torchvision.datasets.ImageFolder(DIR_CARTOON, transform=transf)
    sketch_dataset = torchvision.datasets.ImageFolder(DIR_SKETCH, transform=transf)

    #fix the random seed
    np.random.seed(1)

    n = len(photo_dataset)  # total number of examples
    indices = list(range(n))
    np.random.shuffle(indices)  # shuffle indices
    split = int(0.1 * n)  # take ~10% for test
    train_indices, test_indices = indices[split:], indices[:split]
    photo_testset = torch.utils.data.Subset(photo_dataset, test_indices)  # take first 10%
    photo_trainset = torch.utils.data.Subset(photo_dataset, train_indices)  # take the rest
    photo_trainset.targets = np.array(photo_dataset.targets)[list(train_indices)].tolist()
    photo_testset.targets = np.array(photo_dataset.targets)[list(test_indices)].tolist()

    n = len(art_dataset)
    indices = list(range(n))
    np.random.shuffle(indices)
    split = int(0.1 * n)
    train_indices, test_indices = indices[split:], indices[:split]
    art_testset = torch.utils.data.Subset(art_dataset, test_indices)
    art_trainset = torch.utils.data.Subset(art_dataset, train_indices)
    art_trainset.targets = np.array(art_dataset.targets)[list(train_indices)].tolist()
    art_testset.targets = np.array(art_dataset.targets)[list(test_indices)].tolist()

    n = len(cartoon_dataset)
    indices = list(range(n))
    np.random.shuffle(indices)
    split = int(0.1 * n)
    train_indices, test_indices = indices[split:], indices[:split]
    cartoon_testset = torch.utils.data.Subset(cartoon_dataset, test_indices)
    cartoon_trainset = torch.utils.data.Subset(cartoon_dataset, train_indices)
    cartoon_trainset.targets = np.array(cartoon_dataset.targets)[list(train_indices)].tolist()
    cartoon_testset.targets = np.array(cartoon_dataset.targets)[list(test_indices)].tolist()

    n = len(sketch_dataset)
    indices = list(range(n))
    np.random.shuffle(indices)
    split = int(0.1 * n)
    train_indices, test_indices = indices[split:], indices[:split]
    sketch_testset = torch.utils.data.Subset(sketch_dataset, test_indices)
    sketch_trainset = torch.utils.data.Subset(sketch_dataset, train_indices)
    sketch_trainset.targets = np.array(sketch_dataset.targets)[list(train_indices)].tolist()
    sketch_testset.targets = np.array(sketch_dataset.targets)[list(test_indices)].tolist()

    trainsets = [photo_trainset, art_trainset, cartoon_trainset, sketch_trainset]
    testsets = [photo_testset, art_testset, cartoon_testset, sketch_testset]

    return trainsets, testsets

def get_PACS_dataloaders(args):
    #get dataset
    trainsets, testsets = get_PACS_datasets(random_seed=args.seed)
    datasets_name = ['photo', 'art_painting', 'cartoon', 'sketch']
    datasets_client_index = {'photo':[], 'art_painting':[], 'cartoon':[], 'sketch':[]}
    #get number of classes
    num_classes = 7
    col_names = [f"class{i}" for i in range(num_classes)]
    # print(col_names)
    hist_color = '#4169E1'
    # plt.rcParams['figure.facecolor'] = 'white'

    clients_indexset = [ i for i in range(args.K)]
    clientnumbers_per_dataset = int(args.K/len(datasets_name))
    np.random.seed(args.seed)
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
    for dataset_name, trainset in zip(datasets_name, trainsets):
        dataset_client_num = len(datasets_client_index[dataset_name])
        if 'iid' in args.split:
            labeldir_part = FMNISTPartitioner(trainset.targets, 
                                                        num_clients=dataset_client_num,
                                                        partition="iid",
                                                        seed=1)
        elif '_2' in args.split:
            labeldir_part = PACSPartitioner(trainset.targets, 
                                            num_clients=dataset_client_num,
                                            balance=None, 
                                            partition="shards",
                                            num_shards=dataset_client_num*2,
                                            seed=1)
        elif '_3' in args.split:
            labeldir_part = PACSPartitioner(trainset.targets, 
                                            num_clients=dataset_client_num,
                                            balance=None, 
                                            partition="shards",
                                            num_shards=dataset_client_num*3,
                                            seed=1)
        elif 'dir' in args.split:
            labeldir_part = PACSPartitioner(trainset.targets, 
                                            num_clients=dataset_client_num,
                                            balance=None, 
                                            partition="dirichlet", 
                                            dir_alpha=float(args.split[-3:]),
                                            seed=1)
        else:
            raise ValueError("Wrong split argument")
        
        for i, k in enumerate(datasets_client_index[dataset_name]):
            Dtrs[k] = DataLoader(DatasetSplit(trainsets[clients_dataset_index[k]], labeldir_part.client_dict[i]), batch_size=args.B, shuffle=True)


    return Dtrs
