#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
from torch import nn
import torch.nn.functional as F
import copy 
import numpy as np

 
import copy
import sys
sys.path.append("..") 

from utils.global_test import *
from args import args_parser
 

def aggregation(client_index, global_model, client_models, dict_users, t=None, t_start=None, global_models=None, 
                    period_size=3, window_size=3, agg_mode = 'wa', args=None,testset=None, m_t=None, v_t=None, a_list=None):
    s = 0
    for j in client_index:
        # normal
        s += len(dict_users[j]) #在client.train定义为本地训练数据集的大小

    global_w = global_model.state_dict()
    net_para_keys = global_model.state_dict().keys()

    if agg_mode == 'wa':
        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()

            if i == 0:
                for key in net_para_keys:
                    global_w[key] = net_para[key] * ( len(dict_users[j]) / s)
            else:
                for key in net_para_keys:
                    global_w[key] += net_para[key] * ( len(dict_users[j]) / s)

    elif agg_mode == 'gma' or agg_mode == 'gma+ima':
        server_lr = 0.0001
        p_thresh = 0.8
        #graidient masked averaging
        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()

            if i == 0:
                for key in net_para_keys:
                    global_w[key] = net_para[key] * ( len(dict_users[j]) / s)
            else:
                for key in net_para_keys:
                    global_w[key] += net_para[key] * ( len(dict_users[j]) / s)

        current_global_model = copy.deepcopy(global_model)
        current_global_model.load_state_dict(global_w)
        sign_counter = copy.deepcopy(global_model) 
        for k in sign_counter.state_dict().keys():
            sign_counter.state_dict()[k].data.zero_()

        for index in client_index:
            for p1, p2, p3 in zip(global_model.parameters(), client_models[index].parameters(), sign_counter.parameters()):
                p2_grad_sign = torch.sign(p2-p1)
                p3.data += p2_grad_sign

        for index in client_index:
            for p0, p1, p2, p3 in zip(current_global_model.parameters(), global_model.parameters(), client_models[index].parameters(), sign_counter.parameters()):
                p2_mask = 1 * (p2-p1 > 0)
                p3_mask = 1 * (p3.data > 0)
                final_mask = torch.logical_and(torch.logical_not(torch.logical_xor(p2_mask, p3_mask)), 1 * (torch.abs(p3.data) > p_thresh * len(client_index)))
                new_grad = (p2-p1) * final_mask
                p1.data = p0.data
                p1.data -= (server_lr * new_grad/len(client_index))

        global_w = global_model.state_dict()

        if agg_mode == 'gma+ima':
            if t < window_size:
                global_models[t]=copy.deepcopy(global_w)
            else:
                global_index = t % window_size  
                global_models[global_index] = copy.deepcopy(global_w)
                if t >= t_start :
                    if    len(testset) >10:
                        test_global_model = copy.deepcopy(global_model)
                        test_global_model.load_state_dict(global_w)
                        acc, loss = test_on_globaldataset(args, test_global_model, testset)
                        print(f'test acc:{acc.data}, test loss:{loss} before moving averaging')
                    for i in range(window_size):
                        for key in net_para_keys:
                            if i == 0:
                                global_w[key] = global_models[i][key]/window_size 
                            else:
                                global_w[key] += global_models[i][key]/window_size 

    elif agg_mode == 'fedyogi' or agg_mode == 'fedyogi+ima':
        current_global_w = copy.deepcopy(global_w)
        eta = 1e-2
        eta_l = 0.0316
        beta_1 = 0.9
        beta_2  = 0.99
        tau = 1e-3

        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()

            if i == 0:
                for key in net_para_keys:
                    global_w[key] = net_para[key] * ( len(dict_users[j]) / s)
            else:
                for key in net_para_keys:
                    global_w[key] += net_para[key] * ( len(dict_users[j]) / s)

        
        if agg_mode == 'fedyogi+ima':
            if t < window_size:
                global_models[t]=copy.deepcopy(global_w)
            else:
                global_index = t % window_size  
                global_models[global_index] = copy.deepcopy(global_w)
                if t >= t_start :
                    if t == t_start:
                        m_t, v_t = copy.deepcopy(global_model.state_dict()), copy.deepcopy(global_model.state_dict())
                        for k in m_t.keys():
                            m_t[k].data.zero_()
                            v_t[k].data.zero_()
                    if    len(testset) >10:
                        test_global_model = copy.deepcopy(global_model)
                        test_global_model.load_state_dict(global_w)
                        acc, loss = test_on_globaldataset(args, test_global_model, testset)
                        print(f'test acc:{acc.data}, test loss:{loss} before moving averaging')
                    for i in range(window_size):
                        for key in net_para_keys:
                            if i == 0:
                                global_w[key] = global_models[i][key]/window_size 
                            else:
                                global_w[key] += global_models[i][key]/window_size 

        delta_t = copy.deepcopy(global_w)
        for key in net_para_keys:   
            delta_t[key] = global_w[key] - current_global_w[key]
        # m_t
        for key in m_t.keys(): 
            m_t[key] = beta_1*m_t[key] + (1-beta_1)*delta_t[key]
        # v_t
        for key in net_para_keys: 
            v_t[key] =  v_t[key]  - (1.0 - beta_2)*(delta_t[key]*delta_t[key])*torch.sign(v_t[key]-(delta_t[key]*delta_t[key]))
        # new_weights
        for key in net_para_keys: 
            global_w[key] = current_global_w[key] +   eta * m_t[key] / (torch.sqrt(v_t[key]) + tau)


    elif agg_mode == 'fedadam' or agg_mode == 'fedadam+ima':
        current_global_w = copy.deepcopy(global_w)
        eta = 1e-2
        eta_l = 1e-1
        beta_1 = 0.9
        beta_2  = 0.99
        tau = 1e-3

        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()

            if i == 0:
                for key in net_para_keys:
                    global_w[key] = net_para[key] * ( len(dict_users[j]) / s)
            else:
                for key in net_para_keys:
                    global_w[key] += net_para[key] * ( len(dict_users[j]) / s)

        if agg_mode == 'fedadam+ima':
            if t < window_size:
                global_models[t]=copy.deepcopy(global_w)
            else:
                global_index = t % window_size  
                global_models[global_index] = copy.deepcopy(global_w)
                if t >= t_start:
                    if t == t_start:
                        m_t, v_t = copy.deepcopy(global_model.state_dict()), copy.deepcopy(global_model.state_dict())
                        for k in m_t.keys():
                            m_t[k].data.zero_()
                            v_t[k].data.zero_()
                    if   len(testset) >10:
                        test_global_model = copy.deepcopy(global_model)
                        test_global_model.load_state_dict(global_w)
                        acc, loss = test_on_globaldataset(args, test_global_model, testset)
                        print(f'test acc:{acc.data}, test loss:{loss} before moving averaging')
                    for i in range(window_size):
                        for key in net_para_keys:
                            if i == 0:
                                global_w[key] = global_models[i][key]/window_size 
                            else:
                                global_w[key] += global_models[i][key]/window_size 

        delta_t = copy.deepcopy(global_w)
        for key in net_para_keys:   
            delta_t[key] = global_w[key] - current_global_w[key]
            
        # m_t
        for key in net_para_keys: 
            m_t[key] = beta_1*m_t[key] + (1-beta_1)*delta_t[key]
        # v_t
        for key in net_para_keys: 
            v_t[key] = beta_2*v_t[key]  + (1.0 - beta_2)*(delta_t[key]*delta_t[key])
        # new_weights
        for key in net_para_keys: 
            global_w[key] = current_global_w[key] +   eta * m_t[key] / (torch.sqrt(v_t[key]) + tau)


    elif agg_mode == 'fedadagrad' or agg_mode == 'fedadagrad+ima':
        current_global_w = copy.deepcopy(global_w)
        eta = 1e-1
        eta_l = 1e-1
        beta_1 = 0.0
        beta_2  = 0.0
        tau = 1e-9

        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()

            if i == 0:
                for key in net_para_keys:
                    global_w[key] = net_para[key] * ( len(dict_users[j]) / s)
            else:
                for key in net_para_keys:
                    global_w[key] += net_para[key] * ( len(dict_users[j]) / s)

        if agg_mode == 'fedadagrad+ima':
            if t < window_size:
                global_models[t]=copy.deepcopy(global_w)
            else:
                global_index = t % window_size  
                global_models[global_index] = copy.deepcopy(global_w)
                if t >= t_start :
                    if   len(testset) >10:
                        test_global_model = copy.deepcopy(global_model)
                        test_global_model.load_state_dict(global_w)
                        acc, loss = test_on_globaldataset(args, test_global_model, testset)
                        print(f'test acc:{acc.data}, test loss:{loss} before moving averaging')
                    for i in range(window_size):
                        for key in net_para_keys:
                            if i == 0:
                                global_w[key] = global_models[i][key]/window_size 
                            else:
                                global_w[key] += global_models[i][key]/window_size 

        delta_t = copy.deepcopy(global_w)
        for key in net_para_keys:   
            delta_t[key] = global_w[key] - current_global_w[key]
        # m_t
        for key in net_para_keys: 
            m_t[key] = beta_1*m_t[key] + (1-beta_1)*delta_t[key]
        # v_t
        for key in net_para_keys: 
            v_t[key] =  v_t[key]  +  (delta_t[key]*delta_t[key])
        # new_weights
        for key in net_para_keys: 
            global_w[key] = current_global_w[key] +   eta * m_t[key] / (torch.sqrt(v_t[key]) + tau)

    elif agg_mode =='fednova' or agg_mode == 'fednova+ima':
        d_total_round = copy.deepcopy(global_w)
        # compute d_total_round: the mean of  client nomalized updates
        for i, j in enumerate(client_index):
            d_j = client_models[j]
            if i == 0:
                for key in net_para_keys:
                    d_total_round[key] = d_j[key] * ( len(dict_users[j]) / s)
            else:
                for key in net_para_keys:
                    d_total_round[key] += d_j[key] * ( len(dict_users[j]) / s)

        # update global model
        coeff = 0.0
        for i in client_index:
            coeff = coeff + a_list[i] * ( len(dict_users[i]) / s)
        for key in global_w:
            if global_w[key].type() == 'torch.LongTensor':
                global_w[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
            elif global_w[key].type() == 'torch.cuda.LongTensor':
                global_w[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
            else:
                global_w[key] -= coeff * d_total_round[key]

        if agg_mode == 'fednova+ima':
            if t < window_size:
                global_models[t]=copy.deepcopy(global_w)
            else:
                global_index = t % window_size  
                global_models[global_index] = copy.deepcopy(global_w)
                if t >= t_start:
                    if   len(testset) >10:
                        test_global_model = copy.deepcopy(global_model)
                        test_global_model.load_state_dict(global_w)
                        acc, loss = test_on_globaldataset(args, test_global_model, testset)
                        print(f'test acc:{acc.data}, test loss:{loss} before moving averaging')
                    for i in range(window_size):
                        for key in net_para_keys:
                            if i == 0:
                                global_w[key] = global_models[i][key]/window_size 
                            else:
                                global_w[key] += global_models[i][key]/window_size 



    elif agg_mode == 'ima':
        '''
        t:  current round number
        t_start: the round of performing ima
        window_size: the size of window for iterate averaging
        '''
        accumulated_global_w = copy.deepcopy(global_w)
        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()

            if i == 0:
                for key in net_para_keys:
                    global_w[key] = net_para[key] * ( len(dict_users[j]) / s)
            else:
                for key in net_para_keys:
                    global_w[key] += net_para[key] * ( len(dict_users[j]) / s)    

        if t < window_size:
            global_models[t]=copy.deepcopy(global_w)
        else:
            global_index = t % window_size  
            global_models[global_index] = copy.deepcopy(global_w)
            if t >= t_start:
                if   len(testset) >10:
                    test_global_model = copy.deepcopy(global_model)
                    test_global_model.load_state_dict(global_w)
                    acc, loss = test_on_globaldataset(args, test_global_model, testset)
                    print(f'test acc:{acc.data}, test loss:{loss} before moving averaging')
                for i in range(window_size):
                    for key in net_para_keys:
                        if i == 0:
                            global_w[key] = global_models[i][key]/window_size 
                        else:
                            global_w[key] += global_models[i][key]/window_size 


    global_model.load_state_dict(global_w)

    if 'ima' in agg_mode:
        return global_models

