

import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from get_data import get_dataset,get_val_data,get_test_data,get_ground_truth,get_data_whole_slide
from my_model import CustomTensorDataset,CustomTensorDataset_test,SimSiam,inference,projection_MLP,prediction_MLP
import argparse
from sklearn import linear_model

def main(parser):
    batch_size = 256
    inference_func = inference()
    test_data = get_data_whole_slide(parser.data,256,256,256)
    truth_img,truth_label = get_ground_truth(parser.data,parser.mask,256,256,64,1)
    test_dataset = CustomTensorDataset_test(torch.from_numpy(truth_img),torch.from_numpy(truth_img))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    #model_out = SimSiam()
    model_out = torch.load(parser.model)
    model_out.eval()
    embed_arr = []  
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad(): 
        for idx,img1, img2 in test_dataloader:
            img1  = img1.to(device)
            embed = model_out.get_embed(img1)
            count += 1
            embed_arr = np.append(embed_arr,embed.to('cpu').numpy())
    embed_arr= embed_arr.reshape(-1,embed.size()[1])
    print(embed_arr.shape)
    print('get embedding')
    reg = linear_model.RidgeClassifier()
    reg.fit(embed_arr,truth_label)


    test_dataset = CustomTensorDataset_test(torch.from_numpy(test_data),torch.from_numpy(test_data))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    #model_out = SimSiam()
    model_out = torch.load(parser.model)
    model_out.eval()
    embed_arr = []  
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad(): 
        for idx,img1, img2 in test_dataloader:
            img1  = img1.to(device)
            embed = model_out.get_embed(img1)
            count += 1
            embed_arr = np.append(embed_arr,embed.to('cpu').numpy())
    embed_arr= embed_arr.reshape(-1,embed.size()[1])
    mask = reg.predict(embed_arr)
    img_mask = get_data_whole_slide(parser.data,256,256,256,mask)
    np.save('test',img_mask)
    print(img_mask)
    print('get embedding')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--mask')
    parser.add_argument('--model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = get_parser()
    main(parser)
