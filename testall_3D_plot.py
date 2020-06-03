# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:51:09 2020

@author: mhill
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

data_location = "../results/testall_3D/"

task_flag = 2

if task_flag == 0: 
    data = "sarc_data/"
    data_1 = "sarc"
    labels = ["null","1","2","3","4"]
    conf_matrix = np.load(data_location+data+data_1+"_conf_matrix.npy")
elif task_flag == 1:
    data = "dir_data/"
    data_1 = "dir"
    labels = ["null","1","2","3","4"]
    conf_matrix = np.load(data_location+data+data_1+"_conf_matrix.npy")
elif task_flag == 2:
    data = "cell_diff/"
    data_1 = "cell"
    labels = ["null","adult", "cor4u", "iPSC", "neonatal", "w4ESC"]
    conf_matrix = np.load(data_location+data+data_1+"_diff_conf_matrix.npy")

training_metrics = pd.read_csv(data_location+data+data_1+"_train.csv")
training_metrics.drop(columns=["Unnamed: 0"], inplace=True)
testing_metrics = pd.read_csv(data_location+data+data_1+"_test.csv")
testing_metrics.drop(columns=["Unnamed: 0"], inplace=True)


if task_flag == 0 or task_flag == 1:
    if conf_matrix.shape[0] == 4:
        conf_matrix_df = pd.DataFrame(data=conf_matrix, index=labels[1:], columns=labels[1:])
    elif conf_matrix.shape[0] == 5:
        conf_matrix_df = pd.DataFrame(data=conf_matrix, index=labels, columns=labels)
elif task_flag == 2:
    if conf_matrix.shape[0] == 5:
        conf_matrix_df = pd.DataFrame(data=conf_matrix, index=labels[1:], columns=labels[1:])
    elif conf_matrix.shape[0] == 6:
        conf_matrix_df = pd.DataFrame(data=conf_matrix, index=labels, columns=labels)

# Plot loss
plt.plot(training_metrics['loss'])
plt.plot(training_metrics['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig(data_location+data+data_1+"_loss_plot.png")
plt.close()

# Plot mae
plt.plot(training_metrics['mean_absolute_error'])
plt.plot(training_metrics['val_mean_absolute_error'])
plt.title('Model mean absolute error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig(data_location+data+data_1+"_mae_plot.png")
plt.close()


# Plot training & validation categorical accuracies
plt.plot(training_metrics['categorical_accuracy'])
plt.plot(training_metrics['val_categorical_accuracy'])
plt.title('Model categorical accuracy')
plt.ylabel('Categorical Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig(data_location+data+data_1+"_cat_acc_plot.png")
plt.close()

# plot confusion matrix
sn.heatmap(conf_matrix_df, annot=True, cmap="viridis")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
plt.savefig(data_location+data+data_1+"_conf_matrix.png")
plt.close()
