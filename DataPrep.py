# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:31:10 2020

@author: mhill
"""

from skimage import io
from skimage.transform import resize

from scipy.ndimage.interpolation import rotate

import pandas as pd
import numpy as np
import os

def data_prep(table_location, image_location, task_flag, shape):

    if task_flag == 0:

        max_dim_sarc = np.zeros((1,3))

        adult_s_truth = pd.read_excel(table_location+"adult_sarcomerisation.xlsx")
        adult_s_truth.dropna(axis=1, how='all', inplace=True)
        adult_s_truth.dropna(axis=0, how='any', inplace=True)
        adult_s_truth.drop(["sarcomerisation"], axis=1, inplace=True)
        adult_s_truth.columns = ["Cell ID", "Rating"]
        adult_s_truth['Image'] = np.zeros((len(adult_s_truth.index), 1))
        adult_s_truth = adult_s_truth.to_numpy()

        for i in range(adult_s_truth.shape[0]):
            temp_name = adult_s_truth[i,0]
            temp_im = io.imread(image_location+"adult/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim_sarc[0,j]:
                    max_dim_sarc[0,j] = temp_im.shape[j]

            adult_s_truth[i,2] = temp_im

        print("adult_s done \n")

        cor4u_s_truth = pd.read_excel(table_location+"cor4u_sarcomerisation.xlsx")
        cor4u_s_truth.dropna(axis=1, how='all', inplace=True)
        cor4u_s_truth.drop(["sarcomerisation"], axis=1, inplace=True)
        cor4u_s_truth.columns = ["Cell ID", "Rating"]
        cor4u_s_truth['Image'] = np.zeros((len(cor4u_s_truth.index), 1))
        cor4u_s_truth = cor4u_s_truth.to_numpy()

        for i in range(cor4u_s_truth.shape[0]):
            temp_name = cor4u_s_truth[i,0]
            temp_im = io.imread(image_location+"cor4u/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim_sarc[0,j]:
                    max_dim_sarc[0,j] = temp_im.shape[j]

            cor4u_s_truth[i,2] = temp_im

        print("cor4u_s done \n")

        ipsc_s_truth = pd.read_excel(table_location+"ipsc_sarcomerisation.xlsx")
        ipsc_s_truth.dropna(axis=1, how='all', inplace=True)
        ipsc_s_truth.drop(["sarcomerisation"], axis=1, inplace=True)
        ipsc_s_truth.columns = ["Cell ID", "Rating"]
        ipsc_s_truth['Image'] = np.zeros((len(ipsc_s_truth.index), 1))
        ipsc_s_truth = ipsc_s_truth.to_numpy()

        for i in range(ipsc_s_truth.shape[0]):
            temp_name = ipsc_s_truth[i,0]
            temp_im = io.imread(image_location+"IPSCs/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim_sarc[0,j]:
                    max_dim_sarc[0,j] = temp_im.shape[j]

            ipsc_s_truth[i,2] = temp_im

        print("ipsc_s done \n")

        neonatal_s_truth = pd.read_excel(table_location+"neonatal_sarcomerisation.xlsx")
        neonatal_s_truth.dropna(axis=1, how='all', inplace=True)
        neonatal_s_truth.drop(["sarcomerisation"], axis=1, inplace=True)
        neonatal_s_truth.columns = ["Cell ID", "Rating"]
        neonatal_s_truth['Image'] = np.zeros((len(neonatal_s_truth.index), 1))
        neonatal_s_truth = neonatal_s_truth.to_numpy()

        for i in range(neonatal_s_truth.shape[0]):
            temp_name = neonatal_s_truth[i,0]
            temp_im = io.imread(image_location+"neonatal/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim_sarc[0,j]:
                    max_dim_sarc[0,j] = temp_im.shape[j]

            neonatal_s_truth[i,2] = temp_im

        print("neonatal_s done \n")

        w4esc_s_truth = pd.read_excel(table_location+"w4esc_sarcomerisation.xlsx")
        w4esc_s_truth.dropna(axis=1, how='all', inplace=True)
        w4esc_s_truth.drop(["sarcomerisation"], axis=1, inplace=True)
        w4esc_s_truth.columns = ["Cell ID", "Rating"]
        w4esc_s_truth['Image'] = np.zeros((len(w4esc_s_truth.index), 1))
        w4esc_s_truth = w4esc_s_truth.to_numpy()

        for i in range(w4esc_s_truth.shape[0]):
            temp_name = w4esc_s_truth[i,0]
            temp_im = io.imread(image_location+"w4ESC/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim_sarc[0,j]:
                    max_dim_sarc[0,j] = temp_im.shape[j]

            w4esc_s_truth[i,2] = temp_im

        print("w4esc_s done \n")

        sarc_data = np.concatenate((adult_s_truth, cor4u_s_truth, ipsc_s_truth, neonatal_s_truth, w4esc_s_truth))

        print("sarc done \n")

        return sarc_data, max_dim_sarc

    if task_flag == 1:

        max_dim_dir = np.zeros((1,3))

        adult_d_truth = pd.read_excel(table_location+"adult_directionality.xlsx")
        adult_d_truth.dropna(axis=1, how='all', inplace=True)
        adult_d_truth.dropna(axis=0, how='any', inplace=True)
        adult_d_truth.drop(["direction", "Dispersion", "amount", "goodness"], axis=1, inplace=True)
        adult_d_truth.columns = ["Cell ID", "Rating"]
        adult_d_truth['Image'] = np.zeros((len(adult_d_truth.index), 1))
        adult_d_truth = adult_d_truth.to_numpy()

        for i in range(adult_d_truth.shape[0]):
            temp_name = adult_d_truth[i,0]
            temp_im = io.imread(image_location+"adult/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim_dir[0,j]:
                    max_dim_dir[0,j] = temp_im.shape[j]

            adult_d_truth[i,2] = temp_im

        print("adult_d done \n")

        cor4u_d_truth = pd.read_excel(table_location+"cor4u_directionality.xlsx")
        cor4u_d_truth.dropna(axis=1, how='all', inplace=True)
        cor4u_d_truth.drop(["direction", "Dispersion", "amount", "goodness"], axis=1, inplace=True)
        cor4u_d_truth.columns = ["Cell ID", "Rating"]
        cor4u_d_truth['Image'] = np.zeros((len(cor4u_d_truth.index), 1))
        cor4u_d_truth = cor4u_d_truth.to_numpy()

        for i in range(cor4u_d_truth.shape[0]):
            temp_name = cor4u_d_truth[i,0]
            temp_im = io.imread(image_location+"cor4u/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim_dir[0,j]:
                    max_dim_dir[0,j] = temp_im.shape[j]

            cor4u_d_truth[i,2] = temp_im

        print("cor4u_d done \n")

        ipsc_d_truth = pd.read_excel(table_location+"ipsc_directionality.xlsx")
        ipsc_d_truth.dropna(axis=1, how='all', inplace=True)
        ipsc_d_truth.drop(["direction", "Dispersion", "amount", "goodness"], axis=1, inplace=True)
        ipsc_d_truth.columns = ["Cell ID", "Rating"]
        ipsc_d_truth['Image'] = np.zeros((len(ipsc_d_truth.index), 1))
        ipsc_d_truth = ipsc_d_truth.to_numpy()

        for i in range(ipsc_d_truth.shape[0]):
            temp_name = ipsc_d_truth[i,0]
            temp_im = io.imread(image_location+"IPSCs/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim_dir[0,j]:
                    max_dim_dir[0,j] = temp_im.shape[j]

            ipsc_d_truth[i,2] = temp_im

        print("ipsc_d done \n")

        neonatal_d_truth = pd.read_excel(table_location+"neonatal_directionality.xlsx")
        neonatal_d_truth.dropna(axis=1, how='all', inplace=True)
        neonatal_d_truth.drop(["direction", "Dispersion", "amount", "goodness"], axis=1, inplace=True)
        neonatal_d_truth.columns = ["Cell ID", "Rating"]
        neonatal_d_truth['Image'] = np.zeros((len(neonatal_d_truth.index), 1))
        neonatal_d_truth = neonatal_d_truth.to_numpy()

        for i in range(neonatal_d_truth.shape[0]):
            temp_name = neonatal_d_truth[i,0]
            temp_im = io.imread(image_location+"neonatal/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim_dir[0,j]:
                    max_dim_dir[0,j] = temp_im.shape[j]

            neonatal_d_truth[i,2] = temp_im


        print("neonatal_d done \n")

        dir_data = np.concatenate((adult_d_truth, cor4u_d_truth, ipsc_d_truth, neonatal_d_truth))

        print("dir done \n")

        return dir_data, max_dim_dir

    if task_flag == 2:

        max_dim = np.zeros((1,3))

        adult_list = []
        
        for i in os.listdir(image_location+"adult"):
            if os.path.isfile(os.path.join(image_location+"adult",i)) and i.startswith("adult"):
                adult_list.append(i)
        
        temp = np.zeros((len(adult_list), 3))
        adult_truth = pd.DataFrame(data=temp,columns=["Cell ID","Cell Type","Image"])

        for i in range(len(adult_list)):
            adult_truth.iloc[i,0] = adult_list[i][:-4]
            adult_truth.iloc[i,1] = 1

        adult_truth = adult_truth.to_numpy()

        for i in range(len(adult_list)):
            temp_name = adult_truth[i,0]
            temp_im = io.imread(image_location+"adult/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim[0,j]:
                    max_dim[0,j] = temp_im.shape[j]

            adult_truth[i,2] = temp_im

        print("adult done \n")

        cor4u_list = []
        
        for i in os.listdir(image_location+"cor4u"):
            if os.path.isfile(os.path.join(image_location+"cor4u",i)) and i.startswith("cor4u"):
                cor4u_list.append(i)
        
        temp = np.zeros((len(cor4u_list), 3))
        cor4u_truth = pd.DataFrame(data=temp,columns=["Cell ID","Cell Type","Image"])

        for i in range(len(cor4u_list)):
            cor4u_truth.iloc[i,0] = cor4u_list[i][:-4]
            cor4u_truth.iloc[i,1] = 2

        cor4u_truth = cor4u_truth.to_numpy()

        for i in range(len(cor4u_list)):
            temp_name = cor4u_truth[i,0]
            temp_im = io.imread(image_location+"cor4u/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim[0,j]:
                    max_dim[0,j] = temp_im.shape[j]

            cor4u_truth[i,2] = temp_im

        print("cor4u done \n")

        ipsc_list = []
        
        for i in os.listdir(image_location+"IPSCs"):
            if os.path.isfile(os.path.join(image_location+"IPSCs",i)) and i.startswith("ipsc"):
                ipsc_list.append(i)
        
        temp = np.zeros((len(ipsc_list), 3))
        ipsc_truth = pd.DataFrame(data=temp,columns=["Cell ID","Cell Type","Image"])

        for i in range(len(ipsc_list)):
            ipsc_truth.iloc[i,0] = ipsc_list[i][:-4]
            ipsc_truth.iloc[i,1] = 3

        ipsc_truth = ipsc_truth.to_numpy()

        for i in range(len(ipsc_list)):
            temp_name = ipsc_truth[i,0]
            temp_im = io.imread(image_location+"IPSCs/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim[0,j]:
                    max_dim[0,j] = temp_im.shape[j]

            ipsc_truth[i,2] = temp_im

        print("ipsc done \n")

        neonatal_list = []
        
        for i in os.listdir(image_location+"neonatal"):
            if os.path.isfile(os.path.join(image_location+"neonatal",i)) and i.startswith("neonatal"):
                neonatal_list.append(i)
        
        temp = np.zeros((len(neonatal_list), 3))
        neonatal_truth = pd.DataFrame(data=temp,columns=["Cell ID","Cell Type","Image"])

        for i in range(len(neonatal_list)):
            neonatal_truth.iloc[i,0] = neonatal_list[i][:-4]
            neonatal_truth.iloc[i,1] = 4

        neonatal_truth = neonatal_truth.to_numpy()

        for i in range(len(neonatal_list)):
            temp_name = neonatal_truth[i,0]
            temp_im = io.imread(image_location+"neonatal/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim[0,j]:
                    max_dim[0,j] = temp_im.shape[j]

            neonatal_truth[i,2] = temp_im

        print("neonatal done \n")

        w4esc_list = []
        
        for i in os.listdir(image_location+"w4ESC"):
            if os.path.isfile(os.path.join(image_location+"w4ESC",i)) and i.startswith("w4esc"):
                w4esc_list.append(i)
        
        temp = np.zeros((len(w4esc_list), 3))
        w4esc_truth = pd.DataFrame(data=temp,columns=["Cell ID","Cell Type","Image"])

        for i in range(len(w4esc_list)):
            w4esc_truth.iloc[i,0] = w4esc_list[i][:-4]
            w4esc_truth.iloc[i,1] = 5

        w4esc_truth = w4esc_truth.to_numpy()

        for i in range(len(w4esc_list)):
            temp_name = w4esc_truth[i,0]
            temp_im = io.imread(image_location+"w4ESC/"+temp_name+".tif")
            temp_im = np.transpose(temp_im, (1, 2, 0))
            temp_im = resize(temp_im, shape)

            for j in range(3):
                if temp_im.shape[j] > max_dim[0,j]:
                    max_dim[0,j] = temp_im.shape[j]

            w4esc_truth[i,2] = temp_im

        print("w4esc done \n")

        data = np.concatenate((adult_truth, cor4u_truth, ipsc_truth, neonatal_truth, w4esc_truth))

        print("cell diff done \n")

        return data, max_dim

def split(images , labels, ratio):

    images_train = np.zeros((1, images.shape[1], images.shape[2], images.shape[3], images.shape[4]))
    images_test = images_train
    labels_train = np.zeros((1, labels.shape[1]))
    labels_test = labels_train

    for i in range(labels.shape[1]-1):
            ind = np.argwhere(labels[:,i+1]==1)
            np.random.shuffle(ind)
            temp_labels = labels[ind]
            temp_labels = temp_labels.reshape(temp_labels.shape[0], temp_labels.shape[2])
            temp_images = images[ind]
            temp_images = temp_images.reshape(temp_images.shape[0], temp_images.shape[2], temp_images.shape[3], temp_images.shape[4], temp_images.shape[5])
            split_point = int(ratio*temp_labels.shape[0])
            temp_labels_train = temp_labels[:split_point+1]
            temp_images_train = temp_images[:split_point+1]
            temp_labels_test = temp_labels[split_point+1:]
            temp_images_test = temp_images[split_point+1:]
            labels_train = np.concatenate((labels_train, temp_labels_train))
            labels_test = np.concatenate((labels_test, temp_labels_test))
            images_train = np.concatenate((images_train, temp_images_train))
            images_test = np.concatenate((images_test, temp_images_test))

    images_train = images_train[1:]
    np.random.shuffle(images_train)
    images_test = images_test[1:]
    np.random.shuffle(images_test)
    labels_train = labels_train[1:]
    np.random.shuffle(labels_train)
    labels_test = labels_test[1:]
    np.random.shuffle(labels_test)

    return images_train, labels_train, images_test, labels_test
    
def data_augmentation(images, labels, degree, in_shape, rotation=True, mirror=True):
    
    im_aug = images
    labels_aug = labels
    
    lengths = np.zeros((4,labels.shape[1]-1))
    
    if (rotation == False) & (mirror == False):
        return im_aug, labels_aug
    
    for i in range(labels.shape[1]):
    
        if i > 0:
            temp = labels[labels[:,i]==1]
            lengths[0,i-1] = temp.shape[0]
        
    lengths[1,:] = lengths[0,:] * degree
    lengths[2,:] = np.max(lengths[1,:])
    lengths[3,:] = np.ceil(lengths[2,:] / lengths[0,:])
    lengths = lengths.astype(np.uint16)
    
    for i in range(labels.shape[1]):
    
        if i > 0:
            current_img = images[labels[:,i]==1]
            current_lab = labels[labels[:,i]==1]
    
            for j in range(current_img.shape[0]):
            
                for k in range(lengths[3,i-1]):
                
                    hor_flip_flag = bool(np.round(np.random.random_sample()))
                    vert_flip_flag =  bool(np.round(np.random.random_sample()))
                    degree_of_rotation = np.round(np.random.random_sample()*360)
                
                    if rotation == True:
                        temp = rotate(current_img[j], degree_of_rotation, axes=(0,1))
                        temp = resize(temp, (in_shape[0],in_shape[1],in_shape[2]))
                
                    if (mirror == True) & (hor_flip_flag == True):
                        temp = np.flip(temp, axis=0)
                    if (mirror == True) & (vert_flip_flag == True):
                        temp = np.flip(temp, axis=1)
                
                    temp = temp.reshape(1, in_shape[0], in_shape[1], in_shape[2], 1)
                    temp_labels = current_lab[j].reshape(1, labels.shape[1])
                
                    im_aug = np.concatenate((im_aug, temp), axis=0)
                    labels_aug = np.concatenate((labels_aug, temp_labels), axis=0)
                    
                print("Label "+str(i)+": Image "+str(j+1)+" out of "+str(current_img.shape[0])+" done!\n")
    
    return im_aug, labels_aug