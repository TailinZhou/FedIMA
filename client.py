import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from mixed_digit_dataloader import get_mixed_digit_client_dataloaders
from PACS_dataloader import get_PACS_dataloaders


import utils.optimizer as op

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):#idxs对应的是dataset的索引
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


#class client:
def client_update(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    if args.dataset == 'mixed_digit':
        Dtrs = get_mixed_digit_client_dataloaders(args)
    elif args.dataset == 'pacs':
        Dtrs = get_PACS_dataloaders(args)
    for k in client_index: #k对应的是client的索引
        if args.verbose:
            print('client {} training...'.format(k))
        if args.dataset == 'mixed_digit' or args.dataset == 'pacs':
            Dtr = Dtrs[k]
        else:
            Dtr = DataLoader(DatasetSplit(dataset_train, dict_users[k]), batch_size=args.B, shuffle=True)
        client_models[k], loss = op.fedavg_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k],Dtr)
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict

 

def client_fedsam(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    if args.dataset == 'mixed_digit':
        Dtrs = get_mixed_digit_client_dataloaders(args)
    elif args.dataset == 'pacs':
        Dtrs = get_PACS_dataloaders(args)
    for k in client_index: #k对应的是client的索引
        if args.verbose:
            print('client {} training...'.format(k))
        if args.dataset == 'mixed_digit' or args.dataset == 'domainnet'or args.dataset == 'pacs':
            Dtr = Dtrs[k]
        else:
            Dtr = DataLoader(DatasetSplit(dataset_train, dict_users[k]), batch_size=args.B, shuffle=True)
        client_models[k], loss = op.fedsam_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k],Dtr)
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict



def client_prox_update(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    if args.dataset == 'mixed_digit':
        Dtrs = get_mixed_digit_client_dataloaders(args)
    elif args.dataset == 'pacs':
        Dtrs = get_PACS_dataloaders(args)
    for k in client_index: #k对应的是client的索引
        if args.verbose:
            print('client {} training...'.format(k))
        if args.dataset == 'mixed_digit' or args.dataset == 'pacs':
            Dtr = Dtrs[k]
        else:
            Dtr = DataLoader(DatasetSplit(dataset_train, dict_users[k]), batch_size=args.B, shuffle=True)
        client_models[k], loss = op.fedprox_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k],Dtr)
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, loss_dict
 


def client_fednova(args, client_index, client_models, global_model, global_round, dataset_train, dict_users, loss_dict):  # update nn
    
    a_list = {i:[] for i in client_index}
    d_list = {i:[] for i in client_index}
    n_list = {i:[] for i in client_index}
    if args.dataset == 'mixed_digit':
        Dtrs = get_mixed_digit_client_dataloaders(args)
    elif args.dataset == 'pacs':
        Dtrs = get_PACS_dataloaders(args)
    for k in client_index: #k对应的是client的索引
        if args.verbose:
            print('client {} training...'.format(k))
        if args.dataset == 'mixed_digit'or args.dataset == 'pacs':
            Dtr = Dtrs[k]
        else:
            Dtr = DataLoader(DatasetSplit(dataset_train, dict_users[k]), batch_size=args.B, shuffle=True)
        client_models[k], a_i, d_i, loss = op.fednova_optimizer(args, client_models[k], global_model, global_round, dataset_train, dict_users[k],Dtr)
        # if not args.save_model:
        client_models[k] = []
        torch.cuda.empty_cache()
        a_list[k] = a_i
        d_list[k] = d_i
        n_i = len(dict_users[k])
        n_list[k] = n_i
        loss_dict[k].extend(loss)
        
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
        #print(index_nonselect)
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
    return client_models, a_list, d_list, n_list, loss_dict

 

def client_fedfa_cl(args, client_index, anchorloss_funcs, client_models, global_model, global_round, dataset_train, dict_users, loss_dict, ima_round=None):  # update nn
    if args.dataset == 'mixed_digit':
        Dtrs = get_mixed_digit_client_dataloaders(args)
    elif args.dataset == 'pacs':
        Dtrs = get_PACS_dataloaders(args)
    for k in client_index: #k对应的是client的索引
        if args.verbose:
            print('client {} training...'.format(k))
        if args.dataset == 'mixed_digit' or args.dataset == 'pacs':
            Dtr = Dtrs[k]
        else:
            Dtr = DataLoader(DatasetSplit(dataset_train, dict_users[k]), batch_size=args.B, shuffle=True)
        if ima_round == None:
            ima_round = global_round
        anchorloss_funcs[k], client_models[k], loss = op.fedfa_cl_optimizer(args, anchorloss_funcs[k], client_models[k], global_model, global_round, dataset_train, dict_users[k],Dtr, ima_round)
        loss_dict[k].extend(loss)
    
    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]]*args.E 
        loss_dict[j].extend(loss)               
            
        
    return anchorloss_funcs, client_models, loss_dict

