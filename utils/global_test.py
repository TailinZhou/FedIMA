import copy
from itertools import chain


import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import random

import sys
sys.path.append("..") 
from utils.tSNE import FeatureVisualize

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs): 
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def test_on_globaldataset(args, global_model, dataset_test, mode='test', leaf=False, Dte=None):
    global_model.eval()
    # testing
    test_loss = 0.
    correct = 0.
    assert mode in {'train', 'test'}
    if mode=='test':
        if not leaf:
            Dte = DataLoader(dataset_test, batch_size=args.TB, shuffle=False)
        if len(Dte) == 0:
            Warning('test dataloader is empty, please input the testset of leaf dataset')
    elif mode=='train':
        np.random.seed(0)
        shuffled_indices = np.random.permutation(len(dataset_test))
        val_idx = shuffled_indices[:int(0.2*len(dataset_test))]
        Dte = DataLoader(DatasetSplit(dataset_test, val_idx), batch_size=args.TB, shuffle=False)#
    else:
        print('mode only includes test and train')
        
    l = len(Dte)
    pred = []
    y = []
    for batch_idx, (imgs, labels) in enumerate(Dte):
        torch.cuda.empty_cache()
        with torch.no_grad():
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, log_probs = global_model(imgs)#, restricting_classifier = args.RC, train= False

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
            
    # vis = FeatureVisualize(features.cpu().detach().numpy(), labels.cpu().detach().numpy())
    # vis.plot_tsne(save_eps=False)
    
    test_loss /= len(Dte.dataset)
    accuracy = 100.00 * correct / len(Dte.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(Dte.dataset), accuracy))
    global_model.train()
    return accuracy, test_loss     


def test_on_globaldataset_mixed_digit(args, global_model, dataset_test, dict_test):
    
    dataset_sampling = DatasetSplit(dataset_test, dict_test)
    global_model.eval()
    # testing
    test_loss = 0.
    correct = 0.
    Dte = DataLoader(dataset_sampling, batch_size=args.TB, shuffle=True)

    l = len(Dte)
    pred = []
    y = []
    for batch_idx, (imgs, labels) in enumerate(Dte):
        torch.cuda.empty_cache()
        with torch.no_grad():
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, log_probs = global_model(imgs)#, restricting_classifier = args.RC, train= False

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
            
    # vis = FeatureVisualize(features.cpu().detach().numpy(), labels.cpu().detach().numpy())
    # vis.plot_tsne(save_eps=False)
    
    test_loss /= len(Dte.dataset)
    accuracy = 100.00 * correct / len(Dte.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(Dte.dataset), accuracy))
    global_model.train()
    return accuracy, test_loss  

def verify_feature_consistency(args, client_model, dataset, dict_user_verify, numpy = True):
    client_model.eval()
    # testing
    dataset_verify = DatasetSplit(dataset, dict_user_verify)
    Dte = DataLoader(dataset_verify, batch_size=args.TB, shuffle=False)
 
    for batch_idx, (imgs, labels) in enumerate(Dte):
        torch.cuda.empty_cache()
        with torch.no_grad():
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, _ = client_model(imgs) 
    if numpy:
        #features = F.normalize(features)
        features= features#.cpu().detach().numpy()
    return features  


def globalmodel_test_on_localdataset(args, client_index, global_model, dataset_test, dict_user_test):
    
 
    localtest_loss_dict =  {k:[] for k in client_index}
    accuracy_dict =  {k:[] for k in client_index}
    
    global_model.eval()
    for i in client_index:
        # testing
        test_loss = 0.
        correct = 0.
        Dte = DataLoader(DatasetSplit(dataset_test,dict_user_test[i]), batch_size=args.TB, shuffle=False)#

        l = len(Dte)
        pred = []
        y = []
        for batch_idx, (imgs, labels) in enumerate(Dte):
            torch.cuda.empty_cache()
            with torch.no_grad():
                imgs = imgs.to(args.device)
                labels = labels.type(torch.LongTensor).to(args.device)
                _, log_probs = global_model(imgs) 

                # sum up batch loss
                test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
                # get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(Dte.dataset)
        accuracy = 100.00 * correct / len(Dte.dataset)
        localtest_loss_dict[i].append(test_loss)
        accuracy_dict[i].append(accuracy)
    
    global_model.train()
    return accuracy_dict, localtest_loss_dict     


def localmodels_test_on_localdataset(args, client_index, client_models, dataset_test, dict_user_test):
    
    localtest_loss_dict =  {k:[] for k in client_index}
    accuracy_dict =  {k:[] for k in client_index}
    
    for i in client_index:
        client_models[i].eval()
        # testing
        test_loss = 0.
        correct = 0.
        Dte = DataLoader(DatasetSplit(dataset_test,dict_user_test[i]), batch_size=args.TB, shuffle=False)#

        l = len(Dte)
        pred = []
        y = []
        for batch_idx, (imgs, labels) in enumerate(Dte):
            torch.cuda.empty_cache()
            with torch.no_grad():
                imgs = imgs.to(args.device)
                labels = labels.type(torch.LongTensor).to(args.device)
                _, log_probs = client_models[i](imgs) 

                # sum up batch loss
                test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
                # get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(Dte.dataset)
        accuracy = 100.00 * correct / len(Dte.dataset)
        localtest_loss_dict[i] = test_loss
        accuracy_dict[i] = accuracy
    
        client_models[i].train()
    return accuracy_dict, localtest_loss_dict 


def globalmodel_test_on_specifdataset(args, client_index, global_model, dataset_test, dict_user_test):
    
 
    global_model.eval()
    # testing
    test_loss = 0.
    correct = 0.
    Dte = DataLoader(DatasetSplit(dataset_test,dict_user_test[client_index]), batch_size=args.TB, shuffle=True)

    l = len(Dte)
    pred = []
    y = []
    for batch_idx, (imgs, labels) in enumerate(Dte):
        torch.cuda.empty_cache()
        with torch.no_grad():
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            _, log_probs = global_model(imgs)#, restricting_classifier = args.RC, train= False

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(Dte.dataset)
    accuracy = 100.00 * correct / len(Dte.dataset)
    
#     if args.verbose:
#         print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
#             test_loss, correct, len(Dte.dataset), accuracy))
 
    
    return accuracy, test_loss     