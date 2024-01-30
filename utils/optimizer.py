import copy
import numpy as np
import torch, os, random
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs): 
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



def seed_torch(seed, test = False):
    if test:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
def fedavg_optimizer(args, client_model, global_model, global_round, dataset_train, dict_user,Dtr):
    
    #if set(dataset_train.targets[list(dict_user)].tolist()) != set(range(10)):
    seed_torch(seed=args.seed)
    # Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    elif args.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,
                                    weight_decay=0.001,momentum=args.momentum)#
    #print('training...')
    loss_function  = nn.CrossEntropyLoss().to(args.device)
 
    epoch_loss = []
    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            _, y_preds = client_model(imgs)

            loss = loss_function(y_preds, labels) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.verbose and batch_idx % 6 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return client_model, epoch_loss


def fedprox_optimizer(args, client_model, global_model, global_round, dataset_train, dict_user, Dtr):

    seed_torch(seed=args.seed)
    # Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    #print(len(Dtr))
    
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    elif args.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,
                                    weight_decay=0.001,momentum=args.momentum)
    #print('training...')
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    epoch_loss = []
    epoch_proximal_loss = []
    # delect the gradient leaves of global model
    for name, param in global_model.named_parameters():
        param.requires_grad = False
    for epoch in range(args.E):
        batch_loss = []
        batch_grad = []
        client_model.train()
        proximal_loss_list = []
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            _, y_preds = client_model(imgs)

            # compute proximal_term
            proximal_term = 0.0
            for w, w_t in zip(client_model.parameters(), global_model.parameters()):
                proximal_term += torch.sum(torch.pow(w - w_t, 2))
            proximal_loss = (args.mu / 2) * proximal_term
            
            loss = loss_function(y_preds, labels) + proximal_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.verbose and batch_idx % 6 == 0:
                print('proximal_loss: {}'.format(proximal_loss))
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    for name, param in global_model.named_parameters():
        param.requires_grad = True
    return client_model, epoch_loss




from collections import defaultdict
class ASAM:
    def __init__(self, optimizer, model, rho=0.7, eta=0.2):
        #SAM:rho=0.1, eta=0.0
        #ASAM:rho=0.7, eta=0.2 for CNN
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()


def fedsam_optimizer(args, client_model, global_model, global_round, dataset_train, dict_user,Dtr):
    
    #if set(dataset_train.targets[list(dict_user)].tolist()) != set(range(10)):
    seed_torch(seed=args.seed)
    # Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    elif args.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,
                                    weight_decay=0.001,momentum=args.momentum)#
    #print('training...')
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    # minimizer = SAM(optimizer, client_model)
    if 'CNN' in args.split:
        minimizer = ASAM(optimizer, client_model, rho=0.7, eta=0.2)
    else:
        minimizer = ASAM(optimizer, client_model, rho=0.2, eta=0.05)
    epoch_loss = []
    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)

            # Ascent Step
            _, y_preds = client_model(imgs)
            loss = loss_function(y_preds, labels) 
            loss.backward()
            minimizer.ascent_step()

            # Descent Step
            _, y_preds1 = client_model(imgs)
            loss_function(y_preds1, labels).backward()
            minimizer.descent_step()
            
            if args.verbose and batch_idx % 6 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    return client_model, epoch_loss

    
def fednova_optimizer(args, client_model, global_model, global_round, dataset_train, dict_user, Dtr):

    
    seed_torch(seed=args.seed)
    # Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    elif args.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr, 
                                    weight_decay=0.001,momentum=args.momentum)
    #print('training...')
    loss_function  = nn.CrossEntropyLoss().to(args.device)
    #loss_function = nn.MSELoss().to(args.device)
    epoch_loss = []
    tau = 0
    for epoch in range(args.E):
        batch_loss = []
        batch_grad = []
        client_model.train()
        proximal_loss_list = []
        for batch_idx, (imgs, labels) in enumerate(Dtr):
            
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            _, y_preds = client_model(imgs)
            
            # compute total loss
            loss = loss_function(y_preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tau = tau + 1
            
            if args.verbose and batch_idx % 6 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    a_i = (tau - args.momentum * (1 - pow(args.momentum, tau)) / (1 - args.momentum)) / (1 - args.momentum)
    global_model_para = global_model.state_dict()
    client_model_para = client_model.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        norm_grad[key] = torch.true_divide(global_model_para[key]-client_model_para[key], a_i)

    return client_model, a_i, norm_grad, epoch_loss





def fedfa_cl_optimizer(args, anchorloss_func, client_model, global_model, global_round, dataset_train, dict_user,Dtr, ima_round):

    seed_torch(seed=args.seed)
    # Dtr = DataLoader(DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True)
    if  'mixed_digit' in args.dataset or 'pacs' in args.dataset or 'domainnet' in args.dataset or 'shakespeare' in args.dataset or 'sent140' in args.dataset:
        label_set = set()
        for (imgs, labels) in  Dtr:
            a = labels.data.tolist()
            label_set  = label_set.union(set(list(a)))
    else:
        label_set = set(np.array(dataset_train.targets)[list(dict_user)].tolist())
    misslabel_set = {i for i in range(args.num_classes)} - label_set

    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, ima_round)
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    elif args.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr,  
                                    momentum=args.momentum,weight_decay=0.001) #
        #optimizer_c = torch.optim.SGD(client_model.classifier.parameters(), lr=lr)
    optimizer_c = torch.optim.Adam(client_model.classifier.parameters())

    loss_function  = nn.CrossEntropyLoss().to(args.device)
    anchorloss = anchorloss_func.to(args.device)
    epoch_mean_anchor = copy.deepcopy(anchorloss.anchor.data)
    epoch_loss = []
    
    moving_anchor = copy.deepcopy(anchorloss.anchor.data).to(args.device)
    for epoch in range(args.E):
        batch_loss = []
        client_model.train()
        batch_mean_anchor = torch.zeros_like(anchorloss.anchor.data).to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):

            #predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = client_model(imgs)

            ys = labels.float()
            loss_anchor = anchorloss(features, ys, Lambda = args.lambda_anchor)
            if global_round > 0:
                #feature anchor loss
                loss = loss_function(y_preds, labels)  +  loss_anchor 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # if args.dataset == 'pacs':
                #     r_calibrate = 3*args.r/4
                # else:
                r_calibrate = args.r/4
                # r_calibrate = 0 
                if global_round > r_calibrate:
                    #classifier calibration
                    for name, param in client_model.named_parameters():
                        if "classifier" not in name:
                            param.requires_grad = False

                    C = torch.arange(0,args.num_classes).to(args.device)
                    x_c = copy.deepcopy(anchorloss.anchor.data.detach()).to(args.device)
                    y_c = client_model.classifier(x_c)
                    loss_c = loss_function(y_c, C)

                    optimizer_c.zero_grad()
                    loss_c.backward()
                    optimizer_c.step()

                    for name, param in client_model.named_parameters():
                        param.requires_grad = True
            else: 
                loss = loss_function(y_preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # #memorize class feature anchor  with EWA
            for i in set(labels.tolist()):
                batch_mean_anchor[i] += torch.mean(features[labels==i],dim=0)
                # if global_round > 0:
                #     lambda_momentum = args.momentum_anchor
                # else:
                #     lambda_momentum = 0.5
                # moving_anchor[i] = lambda_momentum*moving_anchor[i] + (1-lambda_momentum)*torch.mean(features[labels==i],dim=0)
            
            if args.verbose and batch_idx % 6 == 0:
                print('loss_anchor: {}'.format(loss_anchor))
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(imgs), len(Dtr.dataset),
                           100. * batch_idx / len(Dtr), loss.item()))
            batch_loss.append(loss.item())

        for i in label_set:
            #compute batch mean anchor according to batch label
            batch_mean_anchor[i] = batch_mean_anchor[i]/(batch_idx+1)
            #compute epoch mean anchor according to batch mean anchor
            lambda_momentum = args.momentum_anchor
            epoch_mean_anchor[i] = lambda_momentum*epoch_mean_anchor[i] + (1-lambda_momentum)*batch_mean_anchor[i]

        #memorize epoch loss
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    anchorloss.anchor.data =  epoch_mean_anchor
    #anchorloss.anchor.data =  moving_anchor

    return anchorloss, client_model, epoch_loss
