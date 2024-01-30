import copy
import numpy as np
import torch, os, random
import torch.nn as nn
import torch.nn.functional as F



class AnchorLoss(nn.Module):
    def __init__(self, cls_num, feature_num, ablation=False):
        super().__init__()
        self.cls_num = cls_num
        self.feature_num = feature_num
        if cls_num > feature_num or ablation:
            self.anchor = nn.Parameter(F.normalize(torch.randn(cls_num, feature_num)), requires_grad=True)
            #self.anchor = nn.Parameter((torch.randn(cls_num, feature_num)), requires_grad=True)
        else:
            I = torch.eye(feature_num,feature_num)
            index = torch.LongTensor(random.sample(range(feature_num), cls_num))
            init = torch.index_select(I, 0, index)
            self.anchor = nn.Parameter(init, requires_grad=True)

        
        
    def forward(self, feature, _target, Lambda = 0.1):
        #feature = F.normalize(feature, dim=0)				
        #self.anchor.data = F.normalize(self.anchor.data)			
        centre = self.anchor.cuda().index_select(dim=0, index=_target.long())
        counter = torch.histc(_target, bins=self.cls_num, min=0, max=self.cls_num-1)

        count = counter[_target.long()]
        centre_dis = feature - centre	
        pow_ = torch.pow(centre_dis, 2)			
        sum_1 = torch.sum(pow_, dim=1)				
        dis_ = torch.div(sum_1, count.float())		
        sum_2 = torch.sum(dis_)/self.cls_num				
        res = Lambda*sum_2   						
        return res
