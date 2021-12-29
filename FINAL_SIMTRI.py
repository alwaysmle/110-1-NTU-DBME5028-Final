#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50,efficientnet_b2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import csv
import pandas as pd
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)


os.environ["CUDA_VISIBLE_DEVICES"]="0"
combine = np.load("/home/alison/alison_1223/combine.npy", allow_pickle=True)



def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[36]:



try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    T.GaussianBlur = GaussianBlur
    
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class SimCLRTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std, s=1):
        image_size = 224 if image_size is None else image_size 
        self.transform = T.Compose([
            T.Lambda(lambda x: x.to(torch.float32)),
            T.Lambda(lambda x:  x/255. ),
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            #T.ToTensor(),
            # We blur the image 50% of the time using a Gaussian kernel. We randomly sample σ ∈ [0.1, 2.0], and the kernel size is set to be 10% of the image height/width.
            #T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        x3 = self.transform(x)
        return x1, x2 ,x3


# In[37]:


class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
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


# In[38]:



batch_size = 64
# Build training dataloader
x = torch.from_numpy(combine)
train_dataset = CustomTensorDataset(x,train=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

#test_data =  torch.from_numpy(test)
#test_dataset = CustomTensorDataset(test_data,train=False)
#test_dataloader = DataLoader(test_dataset, batch_size=300)


# In[44]:



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
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
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
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self, backbone=resnet50(pretrained = True)):
        super().__init__()

        self.backbone = backbone
        #self.projector = projection_MLP(backbone.classifier[1].out_features)
        self.projector = projection_MLP(backbone.fc.out_features)
        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()

        # self.backbone = backbone
        # # self.projector = projection_MLP(backbone.output_dim)
        # self.encoder = nn.Sequential( # f encoder
        #     self.backbone,
        #     # self.projector
        # )
        # self.predictor = prediction_MLP()
    
    def forward(self, x1, x2, x3):

        f, h = self.encoder, self.predictor
        # z1, z2 = f(x1), f(x2)
        z1 = f(x1)
        z2 = f(x2)
        z3 = f(x3)
        p1, p2, p3 = h(z1), h(z2), h(z3)
        L = D(p1, z2) / 2 + D(p2, z1) / 2 + D(p1, z3) / 2 + D(p3, z1) / 2
        return  L
    def get_sim(self, x1, x2):
        f = self.encoder
        z1, z2 = f(x1), f(x2)
        L = D(z1, z2,version='val') / 2
        return L



model = SimSiam()
model.to(device)


# In[46]:


base_lr = 0.05
lr= base_lr *batch_size / 256 
momentum = 0.9
weight_decay =  0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
#set the optimizer function using torch.optim as optim library
#from torch import optim
#optimizer = optim.Adam(model.parameters(),lr = 0.0001)
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# In[47]:


min_val_loss = 100
min_epoch = 0
epochs = 30

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for index, img1,img2,img3 in train_dataloader:
        img1,img2,img3 =  img1.to(device),img2.to(device),img3.to(device)
        optimizer.zero_grad()
        output = model.forward(img1,img2,img3)
        train_loss += output.item()*img1.size(0)
        output.backward()
        optimizer.step()
        #print(train_loss)
    ##scheduler.step()
    train_loss = train_loss/len(train_dataloader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f} \t'.format(epoch, train_loss))


    save_name = 'res_no_nor'+str(epoch)+'.pt'
    torch.save(model,save_name)


# In[ ]:




