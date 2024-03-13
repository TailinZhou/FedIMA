import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import copy
import random, os
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from client import *
 
from utils.aggregator import *
from utils.dispatchor import dispatch
from utils.global_test import *
from utils.local_test import *
from utils.AnchorLoss import  *
from utils.bias_var_covar_analysis import *

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs): 
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def seed_torch(seed, test = True):
    if test:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class Server:

    def __init__(self, args, model, dataset, dict_users):
        seed_torch(args.seed)
        self.args = copy.deepcopy(args)
        self.nn = copy.deepcopy(model)
        self.nns = [[] for i in range(self.args.K)]
        self.p_nns = [[] for i in range(self.args.K)]
        self.cls = []
        #self.cocols = []
        self.contrals = []
        key = [i for i in range(self.args.K)]
        self.loss_dict =  dict((k, [0]) for k in key)
        #self.finetune_loss_dict =  dict((k, [0]) for k in key)
        self.index_dict =  dict((i, []) for i in range(args.r))
        self.dataset = dataset
        self.dict_users = dict_users
            
        self.anchorloss = AnchorLoss(self.args.num_classes, self.args.dims_feature).to(args.device)
        for i in range(self.args.K):  
            temp2 = copy.deepcopy(self.anchorloss)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.cls.append(temp2) 
            
     
    def fedavg(self, testset, dict_users_test, iid=False, agg_mode='wa',similarity=False,
               test_global_model_accuracy = False, clients_dataset_index=None, BVCL_analysis=False):
        # assert agg_mode in {'wa','swa','fedbn','ima', 'gma', 'fedyogi','fedadam','fedadagrad','fedyogi+ima','fedadam+ima','fedadagrad+ima','gma+ima'}
        if 'feature' in self.args.skewness: 
            if 'mixed_digit' in self.args.dataset:
                acc_list_dict = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
                datasets_name = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
            elif 'pacs' in self.args.dataset:
                acc_list_dict = {'photo':[], 'art_painting':[], 'cartoon':[], 'sketch':[]}
                datasets_name = ['photo', 'art_painting', 'cartoon', 'sketch']
        else:
            acc_list = []

        if 'ima' in agg_mode:
            global_nns = [[] for i in range(self.args.window_size)]
        if 'yogi' in agg_mode or 'adam' in agg_mode or 'adagrad' in agg_mode:
            m_t, v_t = copy.deepcopy(self.nn.state_dict()), copy.deepcopy(self.nn.state_dict())
            for k in m_t.keys():
                m_t[k].data.zero_()
                v_t[k].data.zero_()

        tk = tqdm(total=self.args.r)
        for t in range(self.args.r):
            tk.set_description(f'algo:fedavg, agg:{agg_mode}, {t}-th round')
            
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C是client客户端采样率
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # 在K个客户端中采样m个clients
            self.index_dict[t]= index
            # print(index)

            # dispatch
            for k in range(self.args.K):
                if k not in index and self.nns[k] != []:
                    self.nns[k] = self.nns[k].cpu()
                    torch.cuda.empty_cache()
            dispatch(index, self.nn, self.nns)
            
            # # local updating
            if t >= self.args.r_ima and 'ima' in agg_mode:
                t1 = t - self.args.r_ima
                if self.args.decay_mode == 'SD':
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                    self.args.weight_decay = 1.0
                    if t1 % self.args.ima_stage == 0 and t1 != 0:
                        self.args.lr = self.args.lr/2
                elif self.args.decay_mode == 'CLR':
                    self.args.weight_decay = 1.0
                    self.args.lr = self.args.lr_ima
                elif self.args.decay_mode == 'ED':
                    if t1 % self.args.ima_stage == 0 and t1 != 0:
                        self.args.E -= 1
                        if self.args.E < 1:
                            self.args.E = 1
                elif self.args.decay_mode == 'NAD':
                    t1 = t
                elif self.args.decay_mode == 'CD':
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                        lr_max = self.args.lr
                        lr_min = self.args.lr_ima
                    self.args.weight_decay = 1.0
                    if t1%self.args.ima_stage >= self.args.ima_stage/2:
                        self.args.lr = lr_min + (t1%self.args.ima_stage)*(lr_max - lr_min)/(self.args.ima_stage/2)
                    else:
                        self.args.lr = lr_max - (t1%self.args.ima_stage)*(lr_max - lr_min)/(self.args.ima_stage/2)
                else:
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                    self.args.weight_decay = self.args.lr_ima_decay
                self.nns, self.loss_dict = client_update(self.args, index, self.nns, self.nn, t1, self.dataset,  self.dict_users,  self.loss_dict)
            else:
                self.nns, self.loss_dict = client_update(self.args, index, self.nns, self.nn, t, self.dataset,  self.dict_users,  self.loss_dict)
            

            # aggregation
            if agg_mode == 'wa':
                aggregation(index, self.nn, self.nns, self.dict_users)     
            elif agg_mode == 'ima':
                global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                global_models=global_nns, window_size=self.args.window_size, agg_mode='ima',args=self.args,testset=testset)
            elif 'yogi' in agg_mode or 'adam' in agg_mode or 'adagrad' in agg_mode:
                if 'ima' in agg_mode:
                    global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                            global_models=global_nns, window_size=self.args.window_size,
                                            agg_mode=agg_mode, args=self.args,testset=testset, m_t=m_t, v_t=v_t) 
                else:
                    aggregation(index, self.nn, self.nns, self.dict_users, agg_mode=agg_mode, m_t=m_t, v_t=v_t)  
            elif 'gma' in agg_mode:
                if 'ima' in agg_mode:
                    global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                            global_models=global_nns, window_size=self.args.window_size,
                                            agg_mode=agg_mode, args=self.args,testset=testset)
                else:
                    aggregation(index, self.nn, self.nns, self.dict_users, agg_mode='gma')

            else:
                NotImplementedError

            if test_global_model_accuracy:
                if 'feature' in self.args.skewness  :
                    acc = 0.0
                    loss = 0.0
                    for index, testset_per in enumerate(testset):
                        acc1, loss1 = test_on_globaldataset_mixed_digit(self.args, self.nn, testset_per, 
                                                                   dict_users_test[datasets_name[index]])
                        acc_list_dict[datasets_name[index]].append(acc1)
                        acc += acc1/len(datasets_name)
                        loss += loss1/len(datasets_name)
                else:
                    acc, loss = test_on_globaldataset(self.args, self.nn, testset)
                    acc_list.append(acc)
                tk.set_postfix(test_acc=acc.data, test_loss=loss, train_loss=mean_train_loss(self.args,self.loss_dict,t,self.index_dict))

            if not self.args.saveModels:
                self.nns = [[] for i in range(self.args.K)]
                torch.cuda.empty_cache()
            # save the model during the training
            tk.update(1)    
        
        if 'feature' in self.args.skewness:
            acc = acc_list_dict
        else:
            acc = acc_list
        #save local models
        if self.args.saveModels:
            for k in range(self.args.K):
                if self.nns[k]!=[]:
                    path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedavg/seed{self.args.seed}/client{k}_model_fedavg_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
                    torch.save(self.nns[k].state_dict(), path)

         #save global model and acc list, loss list        
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedavg/seed{self.args.seed}/global_model_fedavg_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(self.nn.state_dict(), path)

        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedavg/seed{self.args.seed}/acc_list_fedavg_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(acc, path)
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedavg/seed{self.args.seed}/loss_dict_fedavg_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(self.loss_dict, path)

        if not self.args.saveModels:
            self.nns = [[] for i in range(self.args.K)]
            torch.cuda.empty_cache()
        similarity_dict = None
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, acc

    def fedsam(self, testset, dict_users_test, agg_mode='wa', CKA=False, 
                             similarity=False,test_global_model_accuracy = False):
        # assert agg_mode in {'wa','swa','fedbn','ima','fedyogi','fedadam','fedadagrad','fedyogi+ima','fedadam+ima','fedadagrad+ima'}
        if 'feature' in self.args.skewness: 
            if 'mixed_digit' in self.args.dataset:
                acc_list_dict = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
                datasets_name = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
            elif 'pacs' in self.args.dataset:
                acc_list_dict = {'photo':[], 'art_painting':[], 'cartoon':[], 'sketch':[]}
                datasets_name = ['photo', 'art_painting', 'cartoon', 'sketch']
        else:
            acc_list = []
        if 'ima' in agg_mode:
            global_nns = [[] for i in range(self.args.window_size)]
        if 'yogi' in agg_mode or 'adam' in agg_mode or 'adagrad' in agg_mode:
            m_t, v_t = copy.deepcopy(self.nn.state_dict()), copy.deepcopy(self.nn.state_dict())
            for k in m_t.keys():
                m_t[k].data.zero_()
                v_t[k].data.zero_()
 
 
        tk = tqdm(total=self.args.r)
        for t in range(self.args.r):
            # print('round', t + 1, ':')
            tk.set_description(f'algo:fedsam, agg:{agg_mode}, {t}-th round')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C是client客户端采样率
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # 在K个客户端中采样m个clients
            self.index_dict[t]= index

            # dispatch
            #self.dispatch(index)
            for k in range(self.args.K):
                if k not in index and self.nns[k] != []:
                    self.nns[k] = self.nns[k].cpu()
                    torch.cuda.empty_cache()
            dispatch(index, self.nn, self.nns)
 
            # # local updating
            if t >= self.args.r_ima and 'ima' in agg_mode:
                t1 = t - self.args.r_ima
                if self.args.decay_mode == 'SD':
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                    self.args.weight_decay = 1.0
                    if t1 % self.args.ima_stage == 0 and t1 != 0:
                        self.args.lr = self.args.lr/2
                elif self.args.decay_mode == 'CLR':
                    self.args.weight_decay = 1.0
                    self.args.lr = self.args.lr_ima
                elif self.args.decay_mode == 'ED':
                    if t1 % self.args.ima_stage == 0 and t1 != 0:
                        self.args.E -= 1
                        if self.args.E < 1:
                            self.args.E = 1
                elif self.args.decay_mode == 'NAD':
                    t1 = t
                elif self.args.decay_mode == 'CD':
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                        lr_max = self.args.lr
                        lr_min = self.args.lr_ima
                    self.args.weight_decay = 1.0
                    if t1%self.args.ima_stage >= self.args.ima_stage/2:
                        self.args.lr = lr_min + (t1%self.args.ima_stage)*(lr_max - lr_min)/(self.args.ima_stage/2)
                    else:
                        self.args.lr = lr_max - (t1%self.args.ima_stage)*(lr_max - lr_min)/(self.args.ima_stage/2)
                else:
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                    self.args.weight_decay = self.args.lr_ima_decay
                self.nns, self.loss_dict = client_fedsam(self.args, index, self.nns, self.nn, t1, self.dataset,  self.dict_users,  self.loss_dict)
            else:
                self.nns, self.loss_dict = client_fedsam(self.args, index, self.nns, self.nn, t, self.dataset,  self.dict_users,  self.loss_dict)
 
            # aggregation
            if agg_mode == 'wa':
                aggregation(index, self.nn, self.nns, self.dict_users)     
            elif agg_mode == 'ima':
                global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                global_models=global_nns, window_size=self.args.window_size, agg_mode='ima',args=self.args,testset=testset)
            elif 'yogi' in agg_mode or 'adam' in agg_mode or 'adagrad' in agg_mode:
                if 'ima' in agg_mode:
                    global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                            global_models=global_nns, window_size=self.args.window_size,
                                            agg_mode=agg_mode, args=self.args,testset=testset, m_t=m_t, v_t=v_t) 
                else:
                    aggregation(index, self.nn, self.nns, self.dict_users, agg_mode=agg_mode, m_t=m_t, v_t=v_t)  
            elif 'gma' in agg_mode:
                if 'ima' in agg_mode:
                    global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                            global_models=global_nns, window_size=self.args.window_size,
                                            agg_mode=agg_mode, args=self.args,testset=testset)
                else:
                    aggregation(index, self.nn, self.nns, self.dict_users, agg_mode='gma')
            else:
                NotImplementedError
                        
            if not self.args.saveModels:
                self.nns = [[] for i in range(self.args.K)]
                torch.cuda.empty_cache()
            
            if test_global_model_accuracy:
                if 'feature' in self.args.skewness  :
                    acc = 0.0
                    loss = 0.0
                    for index, testset_per in enumerate(testset):
                        acc1, loss1 = test_on_globaldataset_mixed_digit(self.args, self.nn, testset_per, 
                                                                   dict_users_test[datasets_name[index]])
                        acc_list_dict[datasets_name[index]].append(acc1)
                        acc += acc1/len(datasets_name)
                        loss += loss1/len(datasets_name)
                else:
                    acc, loss = test_on_globaldataset(self.args, self.nn, testset)
                    acc_list.append(acc)
                tk.set_postfix(test_acc=acc.data, test_loss=loss, train_loss=mean_train_loss(self.args,self.loss_dict,t,self.index_dict))
                #tqdm.write(f'test acc:{acc.data}, test loss:{loss}, train loss:{mean_train_loss(self.args,self.loss_dict,t,self.index_dict)}')
            tk.update(1) 

        if 'feature' in self.args.skewness  :
            acc = acc_list_dict
        else:
            acc = acc_list

        #save local models
        if self.args.saveModels:
            for k in range(self.args.K):
                if self.nns[k]!=[]:
                    path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedsam/seed{self.args.seed}/client{k}_model_fedsam_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
                    torch.save(self.nns[k].state_dict(), path)
        #save global model and acc list, loss list        
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedsam/seed{self.args.seed}/global_model_fedsam_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(self.nn.state_dict(), path)
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedsam/seed{self.args.seed}/acc_list_fedsam_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(acc, path)
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedsam/seed{self.args.seed}/loss_dict_fedsam_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(self.loss_dict, path)

        if not self.args.saveModels:
            self.nns = [[] for i in range(self.args.K)]
            torch.cuda.empty_cache()
        similarity_dict = None
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, acc
    
    def fednova(self, testset, dict_users_test, agg_mode='fednova', CKA=False, 
                             similarity=False,test_global_model_accuracy = False):
        # assert agg_mode in {'wa','swa','fedbn','ima','fedyogi','fedadam','fedadagrad','fedyogi+ima','fedadam+ima','fedadagrad+ima'}
        if 'feature' in self.args.skewness: 
            if 'mixed_digit' in self.args.dataset:
                acc_list_dict = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
                datasets_name = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
            elif 'pacs' in self.args.dataset:
                acc_list_dict = {'photo':[], 'art_painting':[], 'cartoon':[], 'sketch':[]}
                datasets_name = ['photo', 'art_painting', 'cartoon', 'sketch']
        else:
            acc_list = []
        similarity_dict = {f'class_{c}_feature':[] for c in range(10)}
        similarity_dict.update(feature=[], classifier=[])

        if 'ima' in agg_mode:
            global_nns = [[] for i in range(self.args.window_size)]
        if 'yogi' in agg_mode or 'adam' in agg_mode or 'adagrad' in agg_mode:
            m_t, v_t = copy.deepcopy(self.nn.state_dict()), copy.deepcopy(self.nn.state_dict())
            for k in m_t.keys():
                m_t[k].data.zero_()
                v_t[k].data.zero_()

        tk = tqdm(total=self.args.r)
        for t in range(self.args.r):
            # print('round', t + 1, ':')
            tk.set_description(f'algo:fednova, agg:{agg_mode}, {t}-th round')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C是client客户端采样率
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # 在K个客户端中采样m个clients
            self.index_dict[t]= index

            # dispatch
            #self.dispatch(index)
            for k in range(self.args.K):
                if k not in index and self.nns[k] != []:
                    self.nns[k] = self.nns[k].cpu()
                    torch.cuda.empty_cache()
            dispatch(index, self.nn, self.nns)
 
            # # local updating            
            if t >= self.args.r_ima and 'ima' in agg_mode:
                t1 = t - self.args.r_ima
                if self.args.decay_mode == 'SD':
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                    self.args.weight_decay = 1.0
                    if t1 % self.args.ima_stage == 0 and t1 != 0:
                        self.args.lr = self.args.lr/2
                elif self.args.decay_mode == 'CLR':
                    self.args.weight_decay = 1.0
                    self.args.lr = self.args.lr_ima
                elif self.args.decay_mode == 'ED':
                    if t1 % self.args.ima_stage == 0 and t1 != 0:
                        self.args.E -= 1
                        if self.args.E < 1:
                            self.args.E = 1
                elif self.args.decay_mode == 'NAD':
                    t1 = t
                elif self.args.decay_mode == 'CD':
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                        lr_max = self.args.lr
                        lr_min = self.args.lr_ima
                    self.args.weight_decay = 1.0
                    if t1%self.args.ima_stage >= self.args.ima_stage/2:
                        self.args.lr = lr_min + (t1%self.args.ima_stage)*(lr_max - lr_min)/(self.args.ima_stage/2)
                    else:
                        self.args.lr = lr_max - (t1%self.args.ima_stage)*(lr_max - lr_min)/(self.args.ima_stage/2)
                else:
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                    self.args.weight_decay = self.args.lr_ima_decay
                self.nns, a_list, d_list, _, self.loss_dict = client_fednova(self.args, index, self.nns, self.nn, t1, 
                                                                            self.dataset,  self.dict_users,  self.loss_dict)
            else:
                self.nns, a_list, d_list, _, self.loss_dict = client_fednova(self.args, index, self.nns, self.nn, t, 
                                                                            self.dataset,  self.dict_users,  self.loss_dict)

            # aggregation
            if agg_mode == 'wa':
                aggregation(index, self.nn, self.nns, self.dict_users)     
            elif agg_mode == 'ima':
                global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                global_models=global_nns, window_size=self.args.window_size, agg_mode='ima',args=self.args,testset=testset)
            elif 'yogi' in agg_mode or 'adam' in agg_mode or 'adagrad' in agg_mode:
                if 'ima' in agg_mode:
                    global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                            global_models=global_nns, window_size=self.args.window_size,
                                            agg_mode=agg_mode, args=self.args,testset=testset, m_t=m_t, v_t=v_t) 
                else:
                    aggregation(index, self.nn, self.nns, self.dict_users, agg_mode=agg_mode, m_t=m_t, v_t=v_t)  
            elif 'fednova' in agg_mode:
                if 'ima' in agg_mode:
                    global_nns = aggregation(index, self.nn, d_list, self.dict_users, t=t, t_start=self.args.r_ima, 
                        global_models=global_nns, window_size=self.args.window_size,
                        agg_mode='fednova+ima', args=self.args, testset=testset, a_list=a_list) 
                else:
                    aggregation(index, self.nn, d_list, self.dict_users, agg_mode='fednova', a_list=a_list)  
            elif 'gma' in agg_mode:
                if 'ima' in agg_mode:
                    global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                            global_models=global_nns, window_size=self.args.window_size,
                                            agg_mode=agg_mode, args=self.args,testset=testset)
                else:
                    aggregation(index, self.nn, self.nns, self.dict_users, agg_mode='gma')
            else:
                NotImplementedError
                        
            if not self.args.saveModels:
                self.nns = [[] for i in range(self.args.K)]
                torch.cuda.empty_cache()
            
            if test_global_model_accuracy:
                if 'feature' in self.args.skewness  :
                    acc = 0.0
                    loss = 0.0
                    for index, testset_per in enumerate(testset):
                        acc1, loss1 = test_on_globaldataset_mixed_digit(self.args, self.nn, testset_per, 
                                                                   dict_users_test[datasets_name[index]])
                        acc_list_dict[datasets_name[index]].append(acc1)
                        acc += acc1/len(datasets_name)
                        loss += loss1/len(datasets_name)
                else:
                    acc, loss = test_on_globaldataset(self.args, self.nn, testset)
                    acc_list.append(acc)
                tk.set_postfix(test_acc=acc.data, test_loss=loss, train_loss=mean_train_loss(self.args,self.loss_dict,t,self.index_dict))
                #tqdm.write(f'test acc:{acc.data}, test loss:{loss}, train loss:{mean_train_loss(self.args,self.loss_dict,t,self.index_dict)}')
            tk.update(1)   
        if 'feature' in self.args.skewness  :
            acc = acc_list_dict
        else:
            acc = acc_list

        #save local models
        if self.args.saveModels:
            for k in range(self.args.K):
                if self.nns[k]!=[]:
                    path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fednova/seed{self.args.seed}/client{k}_model_fednova_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
                    torch.save(self.nns[k].state_dict(), path)

        #save global model and acc list, loss list        
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fednova/seed{self.args.seed}/global_model_fednova_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(self.nn.state_dict(), path)
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fednova/seed{self.args.seed}/acc_list_fednova_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(acc, path)
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fednova/seed{self.args.seed}/loss_dict_fednova_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(self.loss_dict, path)

        if not self.args.saveModels:
            self.nns = [[] for i in range(self.args.K)]
            torch.cuda.empty_cache()
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, acc

    def fedprox(self, testset, dict_users_test, agg_mode='wa', CKA=False, 
                             similarity=False,test_global_model_accuracy = False):
        if 'feature' in self.args.skewness: 
            if 'mixed_digit' in self.args.dataset:
                acc_list_dict = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
                datasets_name = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
            elif 'pacs' in self.args.dataset:
                acc_list_dict = {'photo':[], 'art_painting':[], 'cartoon':[], 'sketch':[]}
                datasets_name = ['photo', 'art_painting', 'cartoon', 'sketch']
        else:
            acc_list = []
        similarity_dict = {f'class_{c}_feature':[] for c in range(10)}
        similarity_dict.update(feature=[], classifier=[])
        if agg_mode=='ima':
            global_nns = [[] for i in range(self.args.window_size)]
        if 'yogi' in agg_mode or 'adam' in agg_mode or 'adagrad' in agg_mode:
            m_t, v_t = copy.deepcopy(self.nn.state_dict()), copy.deepcopy(self.nn.state_dict())
            for k in m_t.keys():
                m_t[k].data.zero_()
                v_t[k].data.zero_()
                
        tk = tqdm(total=self.args.r)
        for t in range(self.args.r):
            # print('round', t + 1, ':')
            tk.set_description(f'algo:fedprox, agg:{agg_mode}, {t}-th round')
            # sampling
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C是client客户端采样率
            index = np.random.choice(range(0, self.args.K), m, replace=False)  # 在K个客户端中采样m个clients
            self.index_dict[t]= index

            # dispatch
            #self.dispatch(index)
            for k in range(self.args.K):
                if k not in index and self.nns[k] != []:
                    self.nns[k] = self.nns[k].cpu()
                    torch.cuda.empty_cache()
            dispatch(index, self.nn, self.nns)
 
            # # local updating
            if t >= self.args.r_ima and 'ima' in agg_mode:
                t1 = t - self.args.r_ima
                if self.args.decay_mode == 'SD':
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                    self.args.weight_decay = 1.0
                    if t1 % self.args.ima_stage == 0 and t1 != 0:
                        self.args.lr = self.args.lr/2
                elif self.args.decay_mode == 'CLR':
                    self.args.weight_decay = 1.0
                    self.args.lr = self.args.lr_ima
                elif self.args.decay_mode == 'ED':
                    if t1 % self.args.ima_stage == 0 and t1 != 0:
                        self.args.E -= 1
                        if self.args.E < 1:
                            self.args.E = 1
                elif self.args.decay_mode == 'NAD':
                    t1 = t
                elif self.args.decay_mode == 'CD':
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                        lr_max = self.args.lr
                        lr_min = self.args.lr_ima
                    self.args.weight_decay = 1.0
                    if t1%self.args.ima_stage >= self.args.ima_stage/2:
                        self.args.lr = lr_min + (t1%self.args.ima_stage)*(lr_max - lr_min)/(self.args.ima_stage/2)
                    else:
                        self.args.lr = lr_max - (t1%self.args.ima_stage)*(lr_max - lr_min)/(self.args.ima_stage/2)
                else:
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                    self.args.weight_decay = self.args.lr_ima_decay
                self.nns, self.loss_dict = client_prox_update(self.args, index, self.nns, self.nn, t1, self.dataset,  self.dict_users,  self.loss_dict)
            else:
                self.nns, self.loss_dict = client_prox_update(self.args, index, self.nns, self.nn, t, self.dataset,  self.dict_users,  self.loss_dict)
 
            # aggregation
            if agg_mode == 'wa':
                aggregation(index, self.nn, self.nns, self.dict_users)     
            elif agg_mode == 'ima':
                global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                global_models=global_nns, window_size=self.args.window_size, agg_mode='ima',args=self.args,testset=testset)
            elif 'yogi' in agg_mode or 'adam' in agg_mode or 'adagrad' in agg_mode:
                if 'ima' in agg_mode:
                    global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                            global_models=global_nns, window_size=self.args.window_size,
                                            agg_mode=agg_mode, args=self.args,testset=testset, m_t=m_t, v_t=v_t) 
                else:
                    aggregation(index, self.nn, self.nns, self.dict_users, agg_mode=agg_mode, m_t=m_t, v_t=v_t)  
            elif 'gma' in agg_mode:
                if 'ima' in agg_mode:
                    global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                            global_models=global_nns, window_size=self.args.window_size,
                                            agg_mode=agg_mode, args=self.args,testset=testset)
                else:
                    aggregation(index, self.nn, self.nns, self.dict_users, agg_mode='gma')

            else:
                NotImplementedError

            if not self.args.saveModels:
                self.nns = [[] for i in range(self.args.K)]
                torch.cuda.empty_cache()
            
            if test_global_model_accuracy:
                if 'feature' in self.args.skewness  :
                    acc = 0.0
                    loss = 0.0
                    for index, testset_per in enumerate(testset):
                        acc1, loss1 = test_on_globaldataset_mixed_digit(self.args, self.nn, testset_per, 
                                                                   dict_users_test[datasets_name[index]])
                        acc_list_dict[datasets_name[index]].append(acc1)
                        acc += acc1/len(datasets_name)
                        loss += loss1/len(datasets_name)
                else:
                    acc, loss = test_on_globaldataset(self.args, self.nn, testset)
                    acc_list.append(acc)
                tk.set_postfix(test_acc=acc.data, test_loss=loss, train_loss=mean_train_loss(self.args,self.loss_dict,t,self.index_dict))
                #tqdm.write(f'test acc:{acc.data}, test loss:{loss}, train loss:{mean_train_loss(self.args,self.loss_dict,t,self.index_dict)}')
            tk.update(1)   
        if 'feature' in self.args.skewness  :
            acc = acc_list_dict
        else:
            acc = acc_list
        #save local models
        if self.args.saveModels:
            for k in range(self.args.K):
                if self.nns[k]!=[]:
                    path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedprox/seed{self.args.seed}/client{k}_model_fedprox_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
                    torch.save(self.nns[k].state_dict(), path)

        #save global model and acc list, loss list        
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedprox/seed{self.args.seed}/global_model_fedprox_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(self.nn.state_dict(), path)
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedprox/seed{self.args.seed}/acc_list_fedprox_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(acc, path)
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedprox/seed{self.args.seed}/loss_dict_fedprox_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(self.loss_dict, path)

        if not self.args.saveModels:
            self.nns = [[] for i in range(self.args.K)]
            torch.cuda.empty_cache()
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, acc
    


    def fedfa_anchorloss(self, testset, dict_users_test, agg_mode='wa', similarity=False, fedbn = False,
                         test_global_model_accuracy=False, clients_dataset_index=None):
        similarity_dict = {f'class_{c}_feature':[] for c in range(10)}
        similarity_dict.update(feature=[], classifier=[])

        if 'feature' in self.args.skewness: 
            if 'mixed_digit' in self.args.dataset:
                acc_list_dict = {'MNIST':[], 'SVHN':[], 'USPS':[], 'SynthDigits':[], 'MNIST-M':[]}
                datasets_name = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
            elif 'pacs' in self.args.dataset:
                acc_list_dict = {'photo':[], 'art_painting':[], 'cartoon':[], 'sketch':[]}
                datasets_name = ['photo', 'art_painting', 'cartoon', 'sketch']
        else:
            acc_list = []
        if agg_mode=='ima':
            global_nns = [[] for i in range(self.args.window_size)]
        if 'yogi' in agg_mode or 'adam' in agg_mode or 'adagrad' in agg_mode:
            m_t, v_t = copy.deepcopy(self.nn.state_dict()), copy.deepcopy(self.nn.state_dict())
            for k in m_t.keys():
                m_t[k].data.zero_()
                v_t[k].data.zero_()

        tk = tqdm(total=self.args.r)
        for t in range(self.args.r):
            # print('round', t + 1, ':')
            tk.set_description(f'algo:fedfa, agg:{agg_mode}, {t}-th round')
            np.random.seed(self.args.seed+t)
            m = np.max([int(self.args.C * self.args.K), 1])#C是client客户端采样率
            index = np.random.choice(range(0, self.args.K), m, replace=False)# 在K个客户端中采样m个clients
            self.index_dict[t]= index

            # dispatch
            for k in range(self.args.K):
                if k not in index and self.nns[k] != []:
                    self.nns[k] = self.nns[k].cpu()
                    torch.cuda.empty_cache()
            dispatch(index, self.nn, self.nns)
            dispatch(index, self.anchorloss, self.cls)

            # # local updating
            if t >= self.args.r_ima and 'ima' in agg_mode:
                t1 = t - self.args.r_ima
                if self.args.decay_mode == 'SD':
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                    self.args.weight_decay = 1.0
                    if t1 % self.args.ima_stage == 0 and t1 != 0:
                        self.args.lr = self.args.lr/2
                elif self.args.decay_mode == 'CLR':
                    self.args.weight_decay = 1.0
                    self.args.lr = self.args.lr_ima
                elif self.args.decay_mode == 'ED':
                    t1 = t
                    if t1 % self.args.ima_stage == 0 and t1 != 0:
                        self.args.E -= 1
                        if self.args.E < 1:
                            self.args.E = 1
                elif self.args.decay_mode == 'NAD':
                    t1 = t
                elif self.args.decay_mode == 'CD':
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                        lr_max = self.args.lr
                        lr_min = self.args.lr_ima
                    self.args.weight_decay = 1.0
                    if t1%self.args.ima_stage >= self.args.ima_stage/2:
                        self.args.lr = lr_min + (t1%self.args.ima_stage)*(lr_max - lr_min)/(self.args.ima_stage/2)
                    else:
                        self.args.lr = lr_max - (t1%self.args.ima_stage)*(lr_max - lr_min)/(self.args.ima_stage/2)
                else:
                    if t == self.args.r_ima:
                        self.args.lr = self.args.lr * pow(self.args.weight_decay, self.args.r_ima)
                    self.args.weight_decay = self.args.lr_ima_decay
                    
                self.cls, self.nns, self.loss_dict  = client_fedfa_cl(self.args,index, self.cls, self.nns, self.nn, t, self.dataset,  self.dict_users, self.loss_dict, ima_round=t1) 
            else:
                self.cls, self.nns, self.loss_dict  = client_fedfa_cl(self.args,index, self.cls, self.nns, self.nn, t, self.dataset,  self.dict_users, self.loss_dict)


            # aggregation
            if agg_mode == 'wa':
                aggregation(index, self.nn, self.nns, self.dict_users)     
            elif agg_mode == 'ima':
                global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                global_models=global_nns, window_size=self.args.window_size, agg_mode='ima',args=self.args,testset=testset)
            elif 'yogi' in agg_mode or 'adam' in agg_mode or 'adagrad' in agg_mode:
                if 'ima' in agg_mode:
                    global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                            global_models=global_nns, window_size=self.args.window_size,
                                            agg_mode=agg_mode, args=self.args,testset=testset, m_t=m_t, v_t=v_t) 
                else:
                    aggregation(index, self.nn, self.nns, self.dict_users, agg_mode=agg_mode, m_t=m_t, v_t=v_t)  
            elif 'gma' in agg_mode:
                if 'ima' in agg_mode:
                    global_nns = aggregation(index, self.nn, self.nns, self.dict_users, t=t, t_start=self.args.r_ima, 
                                            global_models=global_nns, window_size=self.args.window_size,
                                            agg_mode=agg_mode, args=self.args,testset=testset)
                else:
                    aggregation(index, self.nn, self.nns, self.dict_users, agg_mode='gma')
            else:
                NotImplementedError
            # anchor aggregation
            aggregation(index, self.anchorloss, self.cls, self.dict_users)

            if not self.args.saveModels:
                self.nns = [[] for i in range(self.args.K)]
                torch.cuda.empty_cache()

            if test_global_model_accuracy:
                if 'feature' in self.args.skewness  :
                    acc = 0.0
                    loss = 0.0
                    for index, testset_per in enumerate(testset):
                        acc1, loss1 = test_on_globaldataset_mixed_digit(self.args, self.nn, testset_per, 
                                                                dict_users_test[datasets_name[index]])
                        acc_list_dict[datasets_name[index]].append(acc1)
                        acc += acc1/len(datasets_name)
                        loss += loss1/len(datasets_name)
                else:
                    acc, loss = test_on_globaldataset(self.args, self.nn, testset)
                    acc_list.append(acc)
                tk.set_postfix(test_acc=acc.data, test_loss=loss, train_loss=mean_train_loss(self.args,self.loss_dict,t,self.index_dict))
                #tqdm.write(f'test acc:{acc.data}, test loss:{loss}, train loss:{mean_train_loss(self.args,self.loss_dict,t,self.index_dict)}')
            tk.update(1)    

        if 'feature' in self.args.skewness  :
            acc = acc_list_dict
        else:
            acc = acc_list
        #save local models
        if self.args.saveModels:
            for k in range(self.args.K):
                if self.nns[k]!=[]:
                    path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedfa/seed{self.args.seed}/client{k}_model_fedfa_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
                    torch.save(self.nns[k].state_dict(), path)

        #save global model and acc list, loss list        
        if agg_mode == 'fedbn':
            for name in datasets_name:
                path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedavg/seed{self.args.seed}/global_model_{name}_fedavg_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
                torch.save(self.nn[name], path)
        else:
            path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedavg/seed{self.args.seed}/global_model_fedavg_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
            torch.save(self.nn.state_dict(), path)
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedfa/seed{self.args.seed}/acc_list_fedfa_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(acc, path)
        path=f"results/Test/{self.args.skewness} skew/{self.args.dataset}/fedfa/seed{self.args.seed}/loss_dict_fedfa_{agg_mode}_{self.args.setup}client_{self.args.split}class.pt"
        torch.save(self.loss_dict, path)

        if not self.args.saveModels:
            self.nns = [[] for i in range(self.args.K)]
            torch.cuda.empty_cache()
        return self.nn, similarity_dict, self.nns, self.loss_dict, self.index_dict, acc

 
    

# >>>>>>>>>>>>>>>>>>>>
def mean_train_loss(args, loss_dict, round, clients_index):
    global_loss = 0.0

    for j in range(args.E):
        for k in (clients_index[round]):
            epoch = args.E*round+j+1
            global_loss += loss_dict[k][epoch]/len(clients_index[round])

    return global_loss


def params_to_vector(parameters):
    return torch.cat(list(map(lambda x: x.detach().flatten(), parameters)))