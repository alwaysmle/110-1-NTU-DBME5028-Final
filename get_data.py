import csv,glob,imageio,os
import numpy as np
import pandas as pd
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
    return   val_data_set1,val_data_set2,val_label