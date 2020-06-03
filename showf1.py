# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:17:18 2020

@author: User
"""

import numpy as np
import math

conf_matrix = np.load("../Results/testall_3D/cell_diff/cell_diff_conf_matrix.npy")

def compute_recall(matrix, index):
    
    recall = matrix[index,index]/np.sum(matrix[:,index])

    if math.isnan(recall)==True:
        recall = 0
    
    return recall

def compute_precision(matrix, index):
    
    precision = matrix[index,index]/np.sum(matrix[index,:])
    
    if math.isnan(precision)==True:
        precision = 0
    
    return precision

def compute_f1(precision, recall):
    
    f1 = 2*(precision*recall)/(precision+recall)
    
    if math.isnan(f1)==True:
        f1 = 0
    
    return f1

def compute_macro_f1(f1_scores):
    
    return np.sum(f1_scores)/len(f1_scores)

def compute_weighted_f1(f1_scores, matrix):
    
    f1 = 0
    
    for i in range(len(f1_scores)):
        
        f1 += np.sum(matrix[i,:])*f1_scores[i]
        
    return f1/np.sum(np.sum(matrix, axis=1), axis=0)

def compute_micro_f1(matrix):
    
    tp = np.sum(np.diagonal(matrix))
    
    fp = np.sum(np.sum(matrix, axis=1), axis=0)-np.sum(np.diagonal(matrix))
    
    return tp/(tp+fp)

f1_scores = np.zeros((conf_matrix.shape[0], 1))

for i in range(conf_matrix.shape[0]):
    
    temp_rec = compute_recall(conf_matrix, i)
    temp_pre = compute_precision(conf_matrix, i)
    
    f1_scores[i] = compute_f1(temp_pre, temp_rec)
    
print("Macro f1\n")
print(compute_macro_f1(f1_scores))
print("\n\n")

print("weighted f1\n")
print(compute_weighted_f1(f1_scores, conf_matrix))
print("\n\n")

print("micro f1\n")
print(compute_micro_f1(conf_matrix))


