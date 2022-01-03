

import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from get_data import get_dataset,get_val_data,get_test_data
from my_model import CustomTensorDataset,CustomTensorDataset_test,SimSiam,inference,projection_MLP,prediction_MLP
import argparse


def main(parser):
    batch_size = 256
    inference_func = inference()
    test_data,test_id = get_test_data(parser.data)
    v1 = torch.from_numpy(test_data)
    test_dataset = CustomTensorDataset_test(v1,v1)
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
    print('get embedding')
    output_arr = inference_func.get_similarity(parser.data,embed_arr,test_id)
    print('get cosine similarity')
    inference_func.result_csv(parser.data,output_arr)
    print('finish output')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = get_parser()
    main(parser)
