import csv,glob,imageio,os
import numpy as np
import pandas as pd
import cv2
def get_data_whole_slide(wsi_path,width,height,inter,mask=None):
    wsi = glob.glob(wsi_path+'/*')
    wsi1 = cv2.imread(wsi[0],3)#[:,:,0]
    patch_arr = []
    if (mask is not None):
        temp = 0
        wsi1 *= 0
        for i in range(0,wsi1.shape[0]-height,inter):
            for j in np.arange(0,wsi1.shape[1]-width,inter):
                wsi1[i:i+height,j:j+width,:] = mask[temp]
                temp += 1   
        return wsi1  
    for i in range(0,wsi1.shape[0]-height,inter):
        for j in np.arange(0,wsi1.shape[1]-width,inter):
            patch = wsi1[i:i+height,j:j+width]
            patch_arr.append(patch)
    return np.array(patch_arr)
def get_ground_truth(wsi_path,mask_path,width,height,inter,cell_type):
    wsi = glob.glob(wsi_path+'/*')
    mask = glob.glob(mask_path+'/*')
    wsi1 = cv2.imread(wsi[0],3)
    mask1 =  (cv2.imread(mask[0],3) == cell_type)
    patch_arr = []
    patch_ans = []
    type_sum = 0
    no_sum = 0
    for i in range(0,wsi1.shape[0]-height,inter):
        for j in np.arange(0,wsi1.shape[1]-width,inter):
            if(type_sum > 20 and no_sum > 20):
                break
            #print(i,j)
            if(np.sum(mask1[i:i+height,j:j+width,0])/(width*height)>0.9 and type_sum<=20):
                #print('want')
                patch = wsi1[i:i+height,j:j+width]
                patch_arr.append(patch)
                patch_ans.append(1)
                type_sum += 1
            if(np.sum(mask1[i:i+height,j:j+width,0])/(width*height)<0.05 and no_sum<=20):
                #print('no')
                patch = wsi1[i:i+height,j:j+width]
                patch_arr.append(patch)
                patch_ans.append(0)
                no_sum += 1
    return np.array(patch_arr),np.array(patch_ans)
def get_dataset(base_path):
    #read combine dataset
    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path,  'test')
    train_names = glob.glob(os.path.join(train_dir, '*') )
    test_names = glob.glob(os.path.join(test_dir, '*') )
    combine_numpy = []
    combine_name = []
    for image_path in train_names:
        id_name = image_path[-16:-4]
        im = imageio.imread(image_path)
        combine_numpy.append(im)
        combine_name.append(id_name)
        #print(len(combine_numpy))
    for image_path in test_names:
        id_name = image_path[-16:-4]
        im = imageio.imread(image_path)
        combine_numpy.append(im)
        combine_name.append(id_name)
        #print(len(combine_numpy)) 
    combine_numpy = np.array(combine_numpy)
    combine_name = np.array(combine_name)
    print('get training data',combine_numpy.shape,combine_name.shape)
    return combine_numpy,combine_name
    #np.save('combine',combine_numpy)
    #np.save('combine_id',combine_name)
def get_val_data(base_path,combine,combine_name):
    val_csv = os.path.join(base_path,  'validation_ground_truth.csv')
    val_csv_file = pd.read_csv(val_csv,header=None)
    val_label = []
    val_data_set1 = []
    val_data_set2 = []
    for i in val_csv_file.values[1:]:
        img2 = i[0][-12:]
        img1 = i[0][-25:-13]
        val_label.append(i[1]) 
        #print(img1,img2)
        img1_id = np.where(img1==combine_name)
        img2_id = np.where(img2==combine_name)
        val_data_set1.append(combine[img1_id])
        val_data_set2.append(combine[img2_id])
    val_data_set1 = np.array(val_data_set1).squeeze()
    val_data_set2 = np.array(val_data_set2).squeeze()
    val_label = np.array(val_label).astype('int')
    print('get validation data',val_data_set1.shape)
    return   val_data_set1,val_data_set2,val_label

def get_test_data(base_path):
    #read combine dataset
    test_dir = os.path.join(base_path,  'test')
    test_names = glob.glob(os.path.join(test_dir, '*') )
    combine_numpy = []
    combine_name = []
    for image_path in test_names:
        id_name = image_path[-16:-4]
        im = imageio.imread(image_path)
        combine_numpy.append(im)
        combine_name.append(id_name)
        #print(len(combine_numpy)) 
    combine_numpy = np.array(combine_numpy)
    combine_name = np.array(combine_name)
    print('get training data',combine_numpy.shape,combine_name.shape)
    return combine_numpy,combine_name