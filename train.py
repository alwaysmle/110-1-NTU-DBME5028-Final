

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
import get_data,my_model
from get_data import get_dataset
from my_model import CustomTensorDataset,CustomTensorDataset_test
os.environ["CUDA_VISIBLE_DEVICES"]="0"
combine = np.load("/home/alison/alison_1223/combine.npy", allow_pickle=True)
#test = np.load("/home/alison/alison_1223/test.npy", allow_pickle=True)
val_data_set1 = np.load('/home/alison/alison_1223/val_data_set1.npy', allow_pickle=True)
val_data_set2 = np.load('/home/alison/alison_1223/val_data_set2.npy', allow_pickle=True)
val_label = np.load('/home/alison/alison_1223/val_label.npy', allow_pickle=True)

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def main(parser):
    batch_size = 64

    x = torch.from_numpy(get_dataset(parser.data))
    train_dataset = CustomTensorDataset(x,train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_model.SimSiam()
    model.to(device)

    epochs = 10
    base_lr = 0.05
    lr= base_lr *batch_size / 256 
    momentum = 0.9
    weight_decay =  0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    val_data_set1,val_data_set2
    v1 = torch.from_numpy(val_data_set1)
    v2 = torch.from_numpy(val_data_set2)
    test_dataset = CustomTensorDataset_test(v1,v2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    min_val_loss = 100
    min_epoch = 0

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

        model.eval()
        output_arr = []  
        count = 0
        with torch.no_grad(): 
            for idx,img1, img2 in test_dataloader:
                img1, img2, = img1.to(device), img2.to(device)
                loss = model.get_sim(img1, img2)
                count += 1
                output_arr = np.append(output_arr,loss.to('cpu').numpy(),axis=0)
        sim_arr = output_arr
        max_num = 0
        max_ses = 0
        for k in range (-999,1000):
            low =  k/1000
            sim_output=sim_arr.copy()
            count = 0 
            for i in range(len(sim_output)):
                if(sim_output[i]<low):
                    sim_output[i] = 1
                    if (sim_output[i] == val_label[i]):
                        count += 1
                else:
                    sim_output[i] = 0
                    if (sim_output[i] == val_label[i]):
                        count += 1
            if (max_num < count/len(sim_output)):
                max_num = count/len(sim_output)
                max_ses = low
        print(max_ses,max_num)
        save_name = 'DOUBLE'+str(epoch)+'.pt'
        torch.save(model,save_name)
    print(save_name)

def get_parser():
    parser.add_argument('--data')
    return parser

if __name__ == "__main__":
    same_seeds(42)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = get_parser()
    main(parser)
