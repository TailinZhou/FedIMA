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



    elif agg_mode == 'pswa':
        '''
        t: current round number
        t_start: the round of performing swa
        period_size: period of moving average
        '''
        if t-t_start>=0:
            accumulated_round =  (t-t_start) % period_size
        else:
            accumulated_round = -1

        accumulated_global_w = copy.deepcopy(global_w)
        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()

            if i == 0:
                for key in net_para_keys:
                    global_w[key] = net_para[key] * ( len(dict_users[j]) / s)
            else:
                for key in net_para_keys:
                    global_w[key] += net_para[key] * ( len(dict_users[j]) / s)    
        if accumulated_round >= 0:
            if accumulated_round == 0:
                torch.save(global_w,f'fedavg_lr{args.lr}_M{args.momentum}_B{args.B}_{args.K}clients_C{args.C}_{t}th_round_global_model_{args.split}.pt')
            test_global_model = copy.deepcopy(global_model)
            test_global_model.load_state_dict(global_w)
            acc, loss = test_on_globaldataset(args, test_global_model, testset)
            print(f'test acc:{acc.data}, test loss:{loss} before moving averaging')
            for key in net_para_keys:
                    global_w[key] = ((accumulated_round+1)*accumulated_global_w[key]  + global_w[key])/(accumulated_round+2)
                    # global_w[key] = ((accumulated_round)*accumulated_global_w[key]  + global_w[key])/(accumulated_round+1)
                    # global_w[key] = (accumulated_global_w[key]  + global_w[key])/(2)

    elif agg_mode == 'swa':
        '''
        t: current round number
        t_start: the round of performing swa
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

        if t-t_start>=0:
            test_global_model = copy.deepcopy(global_model)
            test_global_model.load_state_dict(global_w)
            acc, loss = test_on_globaldataset(args, test_global_model, testset)
            print(f'test acc:{acc.data}, test loss:{loss} before moving averaging')
            for key in net_para_keys:
                    global_w[key] = ((t)*accumulated_global_w[key]  + global_w[key])/(t+1)

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
                    if t >= args.r-20:
                        torch.save(global_w,f'acc results/global_model_beforeIMA_fedavg_ima_{args.r}r_lr{args.lr}_decay{round(1-args.weight_decay, 4)}_M{args.momentum}_B{args.B}_C{args.C}_fima{args.r_ima}_W{args.window_size}_lrdecay{round(1-args.lr_ima_decay, 4)}_{args.dataset}_{args.K}client_{args.split}split.pt')
                for i in range(window_size):
                    for key in net_para_keys:
                        if i == 0:
                            global_w[key] = global_models[i][key]/window_size 
                        else:
                            global_w[key] += global_models[i][key]/window_size 
        # if t == t_start:                   
        #     global_models = copy.deepcopy(global_w)
        # elif t > t_start:
        #     for key in net_para_keys:
        #         lamma = 0.25
        #         global_models[key]  = (1-lamma)*global_models[key] + lamma*global_w[key]
        #         global_w[key] = global_models[key]

 


    elif agg_mode == 'fedbn':
        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()
            if i == 0:
                for key in net_para:
                    if 'bn' not in key:
                        global_w[key] = net_para[key] * ( len(dict_users[j]) / s)
            else:
                for key in net_para:
                    if 'bn' not in key:
                        global_w[key] += net_para[key] * ( len(dict_users[j]) / s)


    global_model.load_state_dict(global_w)

    if 'ima' in agg_mode:
        return global_models



def aggregation_anchor(client_index, global_anchor, client_anchors, dict_users):
    s = 0
    for j in client_index:
        # normal
        s += len(dict_users[j]) #在client.train定义为本地训练数据集的大小

    global_w = global_anchor.state_dict()
    net_para_keys = global_anchor.state_dict().keys()
    anchor_client_index = {class_i:[] for class_i in range(global_w['anchor'].size()[0])}

    for i, j in enumerate(client_index):
        net_para = client_anchors[j].state_dict()

        if i == 0:
            for key in net_para:
                for class_i in range(global_w[key].size()[0]):
                    if net_para[key][class_i].equal(global_w[key][class_i]):
                        continue
                    global_w[key][class_i] = net_para[key][class_i] * len(dict_users[j]) 
                    anchor_client_index[class_i].append(j) 
        else:
            for key in net_para:
                for class_i in range(global_w[key].size()[0]):
                    if net_para[key][class_i].equal(global_w[key][class_i]):
                        continue
                    global_w[key][class_i] += net_para[key][class_i] * len(dict_users[j]) 
                    anchor_client_index[class_i].append(j) 
                    
    for class_i in range(global_w[key].size()[0]):
        s=0
        for client_index in anchor_client_index[class_i]:
            s += len(dict_users[client_index])
        global_w[key][class_i] /= s
    global_anchor.load_state_dict(global_w)
    
    
def aggregation_spline(client_index, global_model, client_models, dict_users, fedbn = False):
    s = 0
    for j in client_index:
        # normal
        s += len(dict_users[j]) #在client.train定义为本地训练数据集的大小

    global_w = global_model.state_dict()
    net_para_keys = global_model.state_dict().keys()
    net_para_keys_list = []
    
    layernumber = 0
    for key in net_para_keys:
        if 'bn' in key:
            continue
        net_para_keys_list.append(key)
        if 'weight' in key:
            layernumber = layernumber + 1

    layer_agg_index = {i+1: [] for i in range(layernumber)}
    for layer_index in range(layernumber):
        neuron_num = global_model.state_dict()[net_para_keys_list[2*layer_index]].size()[0]
        layer_neuron_sim_min = {i: 1.0 for i in range(neuron_num)}
        layer_agg_index[layer_index+1] = {neuron_index: [] for neuron_index in range(neuron_num)}
        for neuron_index in range(neuron_num):
            
            global_neuron_param_weight = global_model.state_dict()[net_para_keys_list[2*layer_index]][neuron_index].view(1,-1).data
            global_neuron_param_bias = global_model.state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index].view(1,-1).data
            global_neuron_param = torch.cat([global_neuron_param_weight, global_neuron_param_bias],1)
            client_neuron_param = {i: [] for i in client_index}

            #get one neuron parameters of all clients
            for k in client_index:
                client_neuron_param_weight = client_models[k].state_dict()[net_para_keys_list[2*layer_index]][neuron_index].view(1,-1).data
                client_neuron_param_bias = client_models[k].state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index].view(1,-1).data
                client_neuron_param[k] = torch.cat([client_neuron_param_weight,client_neuron_param_bias],1)

            #compute the neuron cos similarity 
            for i, client_i in enumerate(client_index):
                for j, client_j in enumerate(client_index):
                    if i <= j:
                        continue
                    cos_sim_value = torch.cosine_similarity(client_neuron_param[client_i],client_neuron_param[client_j]).item()
                    if cos_sim_value <= layer_neuron_sim_min[neuron_index]:
                        layer_neuron_sim_min[neuron_index]=cos_sim_value
                        #print(layer_index+1)
                        layer_agg_index[layer_index+1][neuron_index]=[client_i,client_j]

    for layer_index in range(layernumber):
       
        neuron_num = global_model.state_dict()[net_para_keys_list[2*layer_index]].size()[0]
        for neuron_index in range(neuron_num):
            
            client_agg0 = layer_agg_index[layer_index+1][neuron_index][0] 
            client_agg1 = layer_agg_index[layer_index+1][neuron_index][1]   

            client_neuron_param_weight0 = client_models[client_agg0].state_dict()[net_para_keys_list[2*layer_index]][neuron_index]
            client_neuron_param_bias0 = client_models[client_agg0].state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index]

            client_neuron_param_weight1 = client_models[client_agg1].state_dict()[net_para_keys_list[2*layer_index]][neuron_index]
            client_neuron_param_bias1 = client_models[client_agg1].state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index]

            global_w[net_para_keys_list[2*layer_index]][neuron_index] =(client_neuron_param_weight0+client_neuron_param_weight1)/2
            global_w[net_para_keys_list[2*layer_index+1]][neuron_index] = (client_neuron_param_bias0+client_neuron_param_bias1)/2

        if layer_index == layernumber-1:
            for i, j in enumerate(client_index):
                net_para = client_models[j].state_dict()

                if i == 0:
                    global_w[net_para_keys_list[2*layer_index]] = net_para[net_para_keys_list[2*layer_index]]*(len(dict_users[j])/s)
                    global_w[net_para_keys_list[2*layer_index+1]]=net_para[net_para_keys_list[2*layer_index+1]]*(len(dict_users[j])/s)
                else:
                    global_w[net_para_keys_list[2*layer_index]]+= net_para[net_para_keys_list[2*layer_index]]*(len(dict_users[j])/s)
                    global_w[net_para_keys_list[2*layer_index+1]] += net_para[net_para_keys_list[2*layer_index+1]]*(len(dict_users[j])/s)
    global_model.load_state_dict(global_w)
    
def aggregation_allspline(client_index, global_model, client_models, dict_users, rau=0.05):
    s = 0
    for j in client_index:
        # normal
        s += len(dict_users[j]) #在client.train定义为本地训练数据集的大小

    global_w = global_model.state_dict()
    net_para_keys = global_model.state_dict().keys()
    net_para_keys_list = []
    
    layernumber = 0
    for key in net_para_keys:
        if 'bn' in key:
            continue
        net_para_keys_list.append(key)
        if 'weight' in key:
            layernumber = layernumber + 1

    layer_agg_index = {i+1: [] for i in range(layernumber)}
    for layer_index in range(layernumber):
        neuron_num = global_model.state_dict()[net_para_keys_list[2*layer_index]].size()[0]
        #layer_neuron_sim_min = {i: 1.0 for i in range(neuron_num)}
        layer_neuron_sim_min = {i: 0.0 for i in range(neuron_num)}
        layer_agg_index[layer_index+1] = {neuron_index: [] for neuron_index in range(neuron_num)}
        for neuron_index in range(neuron_num):
            
            global_neuron_param_weight = global_model.state_dict()[net_para_keys_list[2*layer_index]][neuron_index].view(1,-1).data
            global_neuron_param_bias = global_model.state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index].view(1,-1).data
            global_neuron_param = torch.cat([global_neuron_param_weight, global_neuron_param_bias],1)
            client_neuron_param = {i: [] for i in client_index}

            #get one neuron parameters of all clients
            for k in client_index:
                client_neuron_param_weight = client_models[k].state_dict()[net_para_keys_list[2*layer_index]][neuron_index].view(1,-1).data
                client_neuron_param_bias = client_models[k].state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index].view(1,-1).data
                #client_neuron_param[k] = torch.cat([client_neuron_param_weight,client_neuron_param_bias],1)
                client_neuron_param[k] = [client_neuron_param_weight,client_neuron_param_bias]

            #compute the neuron cos similarity 
            for i, client_i in enumerate(client_index):
                for j, client_j in enumerate(client_index):
                    if i <= j:
                        continue
                    # cos_sim_value = torch.cosine_similarity(client_neuron_param[client_i],client_neuron_param[client_j]).item()
                    # if cos_sim_value <= layer_neuron_sim_min[neuron_index]:
                    #     layer_neuron_sim_min[neuron_index]=cos_sim_value
                    #     #print(layer_index+1)
                    #     layer_agg_index[layer_index+1][neuron_index]=[client_i,client_j]
                    #print(client_neuron_param[client_i][1],client_neuron_param[client_j][1])
                    cos_sim_value = 1-torch.cosine_similarity(client_neuron_param[client_i][0],client_neuron_param[client_j][0]).item() + rau*torch.abs(client_neuron_param[client_i][1]-client_neuron_param[client_j][1]).item()

                    if cos_sim_value >= layer_neuron_sim_min[neuron_index]:
                        layer_neuron_sim_min[neuron_index]=cos_sim_value
                        #print(layer_index+1)
                        layer_agg_index[layer_index+1][neuron_index]=[client_i,client_j]

    for layer_index in range(layernumber):
       
        neuron_num = global_model.state_dict()[net_para_keys_list[2*layer_index]].size()[0]
        for neuron_index in range(neuron_num):
            if layer_agg_index[layer_index+1][neuron_index]==[]:
                print(f'layer_index:{layer_index}, neuron_index:{neuron_index}')
            client_agg0 = layer_agg_index[layer_index+1][neuron_index][0] 
            client_agg1 = layer_agg_index[layer_index+1][neuron_index][1]   

            client_neuron_param_weight0 = client_models[client_agg0].state_dict()[net_para_keys_list[2*layer_index]][neuron_index]
            client_neuron_param_bias0 = client_models[client_agg0].state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index]

            client_neuron_param_weight1 = client_models[client_agg1].state_dict()[net_para_keys_list[2*layer_index]][neuron_index]
            client_neuron_param_bias1 = client_models[client_agg1].state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index]

            global_w[net_para_keys_list[2*layer_index]][neuron_index] =(client_neuron_param_weight0+client_neuron_param_weight1)/2
            global_w[net_para_keys_list[2*layer_index+1]][neuron_index] = (client_neuron_param_bias0+client_neuron_param_bias1)/2

        
    global_model.load_state_dict(global_w)
    
def aggregation_allspline1(client_index, global_model, client_models, dict_users, rau=0.05):
    s = 0
    for j in client_index:
        # normal
        s += len(dict_users[j]) #在client.train定义为本地训练数据集的大小

    global_w = global_model.state_dict()
    global_w1 = global_model.state_dict()
    net_para_keys = global_model.state_dict().keys()
    net_para_keys_list = []
    
    layernumber = 0
    for key in net_para_keys:
        if 'bn' in key:
            continue
        net_para_keys_list.append(key)
        if 'weight' in key:
            layernumber = layernumber + 1

    layer_agg_index = {i+1: [] for i in range(layernumber)}
    layer_agg_index1 = {i+1: [] for i in range(layernumber)}
    for layer_index in range(layernumber):
        neuron_num = global_model.state_dict()[net_para_keys_list[2*layer_index]].size()[0]
        #layer_neuron_sim_min = {i: 1.0 for i in range(neuron_num)}
        layer_neuron_sim_min = {i: 0.0 for i in range(neuron_num)}
        layer_neuron_sim_min1 = {i: 0.0 for i in range(neuron_num)}
        layer_agg_index[layer_index+1] = {neuron_index: [] for neuron_index in range(neuron_num)}
        layer_agg_index1[layer_index+1] = {neuron_index: [] for neuron_index in range(neuron_num)}
        for neuron_index in range(neuron_num):
            
            global_neuron_param_weight = global_model.state_dict()[net_para_keys_list[2*layer_index]][neuron_index].view(1,-1).data
            global_neuron_param_bias = global_model.state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index].view(1,-1).data
            global_neuron_param = torch.cat([global_neuron_param_weight, global_neuron_param_bias],1)
            client_neuron_param = {i: [] for i in client_index}

            #get one neuron parameters of all clients
            for k in client_index:
                client_neuron_param_weight = client_models[k].state_dict()[net_para_keys_list[2*layer_index]][neuron_index].view(1,-1).data
                client_neuron_param_bias = client_models[k].state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index].view(1,-1).data
                #client_neuron_param[k] = torch.cat([client_neuron_param_weight,client_neuron_param_bias],1)
                client_neuron_param[k] = [client_neuron_param_weight,client_neuron_param_bias]

            #compute the neuron cos similarity 
            for i, client_i in enumerate(client_index):
                for j, client_j in enumerate(client_index):
                    if i <= j:
                        continue
                    # cos_sim_value = torch.cosine_similarity(client_neuron_param[client_i],client_neuron_param[client_j]).item()
                    # if cos_sim_value <= layer_neuron_sim_min[neuron_index]:
                    #     layer_neuron_sim_min[neuron_index]=cos_sim_value
                    #     #print(layer_index+1)
                    #     layer_agg_index[layer_index+1][neuron_index]=[client_i,client_j]
                    #print(client_neuron_param[client_i][1],client_neuron_param[client_j][1])
                    
                    cos_sim_value = 1-torch.cosine_similarity(client_neuron_param[client_i][0],client_neuron_param[client_j][0]).item() + rau*torch.abs(client_neuron_param[client_i][1]-client_neuron_param[client_j][1]).item()
                    
                    
                    if cos_sim_value >= layer_neuron_sim_min[neuron_index]:
                        layer_neuron_sim_min[neuron_index]=cos_sim_value
                        #print(layer_index+1)
                        layer_agg_index[layer_index+1][neuron_index]=[client_i, client_j]
                        
            for i, client_i in enumerate(client_index):
                for j, client_j in enumerate(client_index):
                    if i <= j or (client_i in layer_agg_index[layer_index+1][neuron_index] and client_j in layer_agg_index[layer_index+1][neuron_index]):
                        continue
                    # cos_sim_value = torch.cosine_similarity(client_neuron_param[client_i],client_neuron_param[client_j]).item()
                    # if cos_sim_value <= layer_neuron_sim_min[neuron_index]:
                    #     layer_neuron_sim_min[neuron_index]=cos_sim_value
                    #     #print(layer_index+1)
                    #     layer_agg_index[layer_index+1][neuron_index]=[client_i,client_j]
                    #print(client_neuron_param[client_i][1],client_neuron_param[client_j][1])
                    
                    cos_sim_value = 1-torch.cosine_similarity(client_neuron_param[client_i][0],client_neuron_param[client_j][0]).item() + rau*torch.abs(client_neuron_param[client_i][1]-client_neuron_param[client_j][1]).item()
                    
                    
                    if cos_sim_value >= layer_neuron_sim_min1[neuron_index]:
                        layer_neuron_sim_min1[neuron_index]=cos_sim_value
                        layer_agg_index1[layer_index+1][neuron_index]=[client_i, client_j]

    for layer_index in range(layernumber):
       
        neuron_num = global_model.state_dict()[net_para_keys_list[2*layer_index]].size()[0]
        for neuron_index in range(neuron_num):
            if layer_agg_index[layer_index+1][neuron_index]==[]:
                print(f'layer_index:{layer_index}, neuron_index:{neuron_index}')
            client_agg0 = layer_agg_index[layer_index+1][neuron_index][0]
            client_agg1 = layer_agg_index[layer_index+1][neuron_index][1]
            
            client_agg2 = layer_agg_index1[layer_index+1][neuron_index][0]
            client_agg3 = layer_agg_index1[layer_index+1][neuron_index][1]
            

            client_neuron_param_weight0 = client_models[client_agg0].state_dict()[net_para_keys_list[2*layer_index]][neuron_index]
            client_neuron_param_bias0 = client_models[client_agg0].state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index]

            client_neuron_param_weight1 = client_models[client_agg1].state_dict()[net_para_keys_list[2*layer_index]][neuron_index]
            client_neuron_param_bias1 = client_models[client_agg1].state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index]

            client_neuron_param_weight2 = client_models[client_agg2].state_dict()[net_para_keys_list[2*layer_index]][neuron_index]
            client_neuron_param_bias2 = client_models[client_agg2].state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index]

            client_neuron_param_weight3 = client_models[client_agg3].state_dict()[net_para_keys_list[2*layer_index]][neuron_index]
            client_neuron_param_bias3 = client_models[client_agg3].state_dict()[net_para_keys_list[2*layer_index+1]][neuron_index]

            
            global_w[net_para_keys_list[2*layer_index]][neuron_index] =(client_neuron_param_weight0+client_neuron_param_weight1+client_neuron_param_weight2+client_neuron_param_weight3)/4
            global_w[net_para_keys_list[2*layer_index+1]][neuron_index] = (client_neuron_param_bias0+client_neuron_param_bias1+client_neuron_param_bias2+client_neuron_param_bias3)/4


    # for key in net_para_keys:
    #     global_w[key] = (t*global_w1[key] + global_w[key])/(t+1)
        #global_w[key] = 0.5*global_w1[key] +  0.5*global_w[key]
    global_model.load_state_dict(global_w)
