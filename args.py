
import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    
    # federated argumentss
    parser.add_argument('--E', type=int, default=5, help='number of rounds of local training')
    parser.add_argument('--r', type=int, default=300, help='number of communication rounds')
    parser.add_argument('--K', type=int, default=100, help='number of total clients') 
    parser.add_argument('--B', type=int, default=50, help='local batch size') 
    parser.add_argument('--TB', type=int, default=1000, help="test batch size")
    parser.add_argument('--C', type=float, default=0.1, help='client samspling rate')

    parser.add_argument('--r_ima', type=int, default=225, help='start round of moving average')
    parser.add_argument('--decay_mode', type=str, default='EXD', help='lr d ecay mode')
    #SD:Stage decay, CLR:constant LR, ED:epoch decay, CD:Cyclic decay, NAD:Non additional decay, EXD:Exponential decay
    parser.add_argument('--ima_stage', type=float, default=20, help='decay stage of lr')
    parser.add_argument('--lr_ima', type=float, default=0.00005, help='learning rate after FedIMA')
    parser.add_argument('--lr_ima_decay', type=float, default=0.97, help='learning rate after FedIMA')
    parser.add_argument('--window_size', type=int, default=5, help='the size of window for iterate averaging') 

    # optimizer arguments
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.99, help='learning rate decay per global round')
    parser.add_argument('--mu', type=float, default=0.01, help='proximal term constant for fedprox (default: 0.1)')
    parser.add_argument('--alph', type=float, default=0.01, help='proximal term constant for feddyn')
    parser.add_argument('--lambda_anchor', type=float, default=0.1, help='anchor proximal term constant')
    parser.add_argument('--tau', type=float, default=0.5, help='moon temperature parameter')
    parser.add_argument('--optimizer', type=str, default='sgd', help='type of optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument('--momentum_anchor', type=float, default=0.5, help="dynamic momentum update for feature anchor(default: 0.5)")

    # model and data split arguments
    parser.add_argument('--dims_feature', type=int, default=128, help="feature dimension") 
    parser.add_argument('--trainset_sample_rate', type=int, default=1, help="trainset sample rate")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes") 
    parser.add_argument('--num_perclass', type=int, default=100, help="number of per class in one client dataset")
    parser.add_argument('--dataset', type=str, default='fmnist', help='dataset name')
    parser.add_argument('--model_name', type=str, default='fmnist', help='model_name')
    parser.add_argument('--split', type=str, default='2CNN_2', help='dataset spliting setting')
    parser.add_argument('--setup', type=str, default='', help='setup of FL')
    parser.add_argument('--skewness', type=str, default='label', help='setup of FL skewness')

    # other arguments
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 1)')
    parser.add_argument('--saveModels', action='store_true', help='save client local models')
    #store_false
    args = parser.parse_args(args=[])

    return args

