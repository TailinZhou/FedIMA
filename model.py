import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
 
import torchvision.models as models
from resnetcifar import ResNet18_cifar10, ResNet34_cifar10, ResNet50_cifar10



class Client_Model(nn.Module):
    def __init__(self, args, name, BFN=False, Smart_init=False):
        super().__init__()
        self.args = args
        self.name = name
        self.BFN = BFN
        self.Smart_init = Smart_init
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)

            
        if self.name == 'emnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, args.dims_feature)#args.dims_feature=100
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
         
        if self.name == 'mnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, args.dims_feature)#args.dims_feature=200
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
            
        if self.name == 'fmnist':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(32*4*4, 384) 
            self.fc2 = nn.Linear(384, args.dims_feature) #args.dims_feature=100
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)

            
        if self.name == 'cifar10':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, args.dims_feature) #args.dims_feature=192
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
            
            
        if self.name == 'mixed_digit':#without BN
            self.n_cls = 10
            self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
            self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
            self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(1152, 384)
            self.fc2 = nn.Linear(384, args.dims_feature) #args.dims_feature=100
            self.classifier = nn.Linear(args.dims_feature, self.n_cls) 

        if self.name == 'mixed_digit_gn':#replace bn with gn
            self.n_cls = 10
            self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
            self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
            self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.gn1 =  nn.GroupNorm(num_groups = 2, num_channels = 64)
            self.gn2 =  nn.GroupNorm(num_groups = 2, num_channels = 64)
            self.gn3 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            self.fc1 = nn.Linear(1152, 384)
            self.fc2 = nn.Linear(384, args.dims_feature) #args.dims_feature=100
            self.classifier = nn.Linear(args.dims_feature, self.n_cls) 

        if self.name == 'cifar100':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
            self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
            self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(2048, 512)
            self.fc2 = nn.Linear(512, args.dims_feature) #args.dims_feature=192
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
            
            
        if self.name == "alexnet_gn":
            self.alexnet = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), 
                nn.GroupNorm(num_groups = 32, num_channels = 64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2), 
                nn.GroupNorm(num_groups = 32, num_channels = 192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1), 
                nn.GroupNorm(num_groups = 32, num_channels = 384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1), 
                nn.GroupNorm(num_groups = 32, num_channels = 256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), 
                nn.GroupNorm(num_groups = 32, num_channels = 256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.AdaptiveAvgPool2d((6, 6)),
                nn.Flatten(),
                nn.Linear(256 * 6 * 6, 1024), 
                nn.ReLU(inplace=True),
                # nn.Dropout(p=0.5),
                nn.Linear(1024, args.dims_feature), 
                nn.ReLU(inplace=True),
                # nn.Dropout(p=0.5)
                )
            self.classifier = nn.Linear(args.dims_feature, args.num_classes)

        if self.name == 'vgg11_gn':
            self.features = nn.Sequential(

                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups = 2, num_channels = 64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups = 2, num_channels = 128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups = 2, num_channels = 256),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups = 2, num_channels = 256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups = 2, num_channels = 512),
                nn.ReLU(inplace=True),

                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups = 2, num_channels = 512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups = 2, num_channels = 512),
                nn.ReLU(inplace=True),

                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups = 2, num_channels = 512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fc = nn.Linear(512, self.args.dims_feature)
            self.classifier = nn.Linear(self.args.dims_feature, self.args.num_classes)

 
        if self.name == 'resnet18':
            resnet18 = ResNet18_cifar10()
            self.model = resnet18
            #del resnet18.fc  
            self.model.fc = nn.Linear(512, self.args.dims_feature) 
            self.classifier = nn.Linear(self.args.dims_feature, self.args.num_classes)

        if self.name == 'resnet18_gn32':
            self.model = ResNet18_cifar10(num_groups_gn=32)
            #del resnet18.fc  
            self.model.fc = nn.Linear(512, self.args.dims_feature) 
            self.classifier = nn.Linear(self.args.dims_feature, self.args.num_classes)


    def forward(self, x):

        if self.name == 'Linear':
            x = self.fc(x)
        
        if self.name == 'mnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            #y_feature = F.relu(self.fc2(x))  
            y_feature = self.fc2(x)
            x = self.classifier(y_feature)
        
        if self.name == 'emnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            x = self.classifier(y_feature)
            
        if self.name == 'fmnist':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32*4*4)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            #y_feature = (self.fc2(x))
            x = self.classifier(y_feature)
            
            
        if self.name == 'cifar10':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            #y_feature = self.fc2(x)
            x = self.classifier(y_feature)
            

        if self.name == 'mixed_digit_gn':
            x = self.pool(F.relu(self.gn1(self.conv1(x))))
            x = self.pool(F.relu(self.gn2(self.conv2(x))))
            x = self.pool(F.relu(self.gn3(self.conv3(x))))
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            #y_feature = F.relu(self.fc2(x))
            y_feature = self.fc2(x)
            x = self.classifier(y_feature)


        if self.name == 'cifar100':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            x = self.classifier(y_feature)


        if self.name == "alexnet_gn":
            y_feature = self.alexnet(x)
            x = self.classifier(y_feature)

            
        if self.name == "vgg11_gn":
            x = self.features(x)
            x = x.view(x.size(0), -1)
            y_feature = self.fc(x)
            x = self.classifier(y_feature)
        

        if self.name == 'resnet18':
            y_feature =  self.model(x)
            x = self.classifier(y_feature)

        if self.name == 'resnet18_gn32':
            y_feature =  self.model(x)
            x = self.classifier(y_feature)

        return y_feature, x

        