# Most of the model is modified from the SimTriplet, https://github.com/hrlblab/SimTriplet
# Line 17-41 75-161
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50,efficientnet_b2,resnet34
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import csv,glob,imageio
import pandas as pd
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    T.GaussianBlur = GaussianBlur
    
class SimCLRTransform():
    def __init__(self, image_size, s=0.8):
        image_size = 224 if image_size is None else image_size 
        self.transform = T.Compose([
            T.Lambda(lambda x: x.to(torch.float32)),
            T.Lambda(lambda x:  x/255. ),
            T.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
            T.RandomGrayscale(p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        x3 = self.transform(x)
        return x1, x2 ,x3

class CustomTensorDataset(TensorDataset):
    def __init__(self, tensors,train):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        self.transforms = SimCLRTransform(image_size=256)
        
    def __getitem__(self, index):
        x = self.tensors[index]
        x1,x2,x3 = self.transforms(x)

        return index,x1,x2,x3

    def __len__(self):
        return len(self.tensors)

class CustomTensorDataset(TensorDataset):
    def __init__(self, tensors,train):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        self.transforms = SimCLRTransform(image_size=256)
        
    def __getitem__(self, index):
        x = self.tensors[index]
        x1,x2,x3 = self.transforms(x)

        return index,x1,x2,x3

    def __len__(self):
        return len(self.tensors)

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    elif version == 'val':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1)
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        if(backbone=='resnet34'):
            self.backbone = resnet34(pretrained = True)
        if(backbone=='resnet50'):
            self.backbone = resnet50(pretrained = True)
        self.projector = projection_MLP(self.backbone.fc.out_features)
        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2, x3):

        f, h = self.encoder, self.predictor
        # z1, z2 = f(x1), f(x2)
        z1 = f(x1)
        z2 = f(x2)
        z3 = f(x3)
        p1, p2, p3 = h(z1), h(z2), h(z3)
        L = D(p1, z2) / 2 + D(p2, z1) / 2  # + D(p1, z3) / 2 + D(p3, z1) / 2
        return  L
    def get_sim(self, x1, x2):
        f = self.encoder
        z1, z2 = f(x1), f(x2)
        L = D(z1, z2,version='val') / 2
        return L
    def get_embed(self, x1):
        f = self.encoder
        return f(x1)





class SimCLRTransform_test():
    def __init__(self, image_size,  s=1.0):
        image_size = 224 if image_size is None else image_size 
        self.transform = T.Compose([
            T.Lambda(lambda x: x.to(torch.float32)),
            T.Lambda(lambda x:  x/255. ),
            #T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        return x1
      
class CustomTensorDataset_test(TensorDataset):
    def __init__(self, tensors1,tensors2):
        self.tensors1 = tensors1
        self.tensors2 = tensors2
        if tensors2.shape[-1] == 3:
            self.tensors1 = tensors1.permute(0, 3, 1, 2)
            self.tensors2 = tensors2.permute(0, 3, 1, 2)
        self.transforms = SimCLRTransform_test(image_size=256)
        
    def __getitem__(self, index):
        x1 = self.tensors1[index]
        x1 = self.transforms(x1)
        x2 = self.tensors2[index]
        x2 = self.transforms(x2)
        return index,x1,x2

    def __len__(self):
        return len(self.tensors2)
class inference():
    def getSimilarityMatrix(self,a1,a2):
        cos_sim = -np.dot(a1,a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
        return cos_sim

    def get_similarity(self,base_path,embed_arr,test_id):
        query_csv = os.path.join(base_path,  'queries.csv')
        query_csv_file = pd.read_csv(query_csv,header=None)
        output_arr = []  
        for i in query_csv_file.values:
            img1 = i[0][:-4]
            img2 = i[1][:-4]
            out1 = np.where(img1==test_id)[0][0]
            out2 = np.where(img2==test_id)[0][0]
            output_arr =  np.append(output_arr,self.getSimilarityMatrix(embed_arr[out1],embed_arr[out2]))
        return output_arr
    def result_csv(self,base_path,output_arr):
        thres = -0.275
        count = 0 
        query_csv = os.path.join(base_path,  'queries.csv')
        count = 0 
        sim_output=output_arr.copy()
        for i in range(len(sim_output)):
            if(sim_output[i]<thres):
                sim_output[i] = 1
            else:
                sim_output[i] = 0
                count += 1
        print(thres,"0 percent",count/len(sim_output))
        csv_name = base_path+'result.csv'
        count = 0
        query_csv_file = pd.read_csv(query_csv,header=None)
        with open(csv_name, 'w', newline='') as result:
            writer = csv.writer(result)
            writer.writerow(['query', 'prediction'])
            for i in query_csv_file.values:
                writer.writerow([i[0][:-4]+'_'+i[1][:-4], int(sim_output[count])])
                count+=1
