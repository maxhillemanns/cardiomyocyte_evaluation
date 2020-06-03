# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:51:09 2020

@author: mhill
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

data_location = "../results/"

task_flag = 0
run = 1

if task_flag == 0:
    data_set = "sarc_data"
    data_set_1 = "sarc"
    labels = ["null","1","2","3","4"]
elif task_flag == 1:
    data_set = "dir_data"
    data_set_1 = "dir"
    labels = ["null","1","2","3","4"]
elif task_flag == 2:
    data_set = "cell_diff"
    data_set_1 = "cell_diff"
    labels = ["null","adult", "cor4u", "iPSC", "neonatal", "w4ESC"]

metrics = ["loss", "mean absolute error", "accuracy", "categorical accuracy"]

## Last layers ##

training_metrics = pd.read_csv(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_training_metrics_some_"+str(run)+".csv")
training_metrics.drop(columns=["Unnamed: 0"], inplace=True)
testing_metrics = pd.read_csv(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_testing_metrics_some_"+str(run)+".csv")
testing_metrics.drop(columns=["Unnamed: 0"], inplace=True)
conf_matrix = np.load(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_conf_matrix_some_"+str(run)+".npy")

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
plt.title('Model loss - last layers trained')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_loss_plot_some_"+str(run)+".png")
plt.close()

# Plot mae
plt.plot(training_metrics['mean_absolute_error'])
plt.plot(training_metrics['val_mean_absolute_error'])
plt.title('Model mean absolute error - last layers trained')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_mae_plot_some_"+str(run)+".png")
plt.close()

# Plot training & validation accuracies
plt.plot(training_metrics['accuracy'])
plt.plot(training_metrics['val_accuracy'])
plt.title('Model accuracy - last layers trained')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_acc_plot_some_"+str(run)+".png")
plt.close()

# Plot training & validation categorical accuracies
plt.plot(training_metrics['categorical_accuracy'])
plt.plot(training_metrics['val_categorical_accuracy'])
plt.title('Model categorical accuracy - last layers trained')
plt.ylabel('Categorical Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_cat_acc_plot_some_"+str(run)+".png")
plt.close()

# plot confusion matrix
sn.heatmap(conf_matrix_df, annot=True, cmap="viridis")
plt.title("Confusion Matrix - last layers trained")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
plt.savefig(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_conf_matrix_some_"+str(run)+".png")
plt.close()


## full network

training_metrics = pd.read_csv(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_training_metrics_all_"+str(run)+".csv")
training_metrics.drop(columns=["Unnamed: 0"], inplace=True)
testing_metrics = pd.read_csv(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_testing_metrics_all_"+str(run)+".csv")
testing_metrics.drop(columns=["Unnamed: 0"], inplace=True)
conf_matrix = np.load(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_conf_matrix_all_"+str(run)+".npy")

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
plt.title('Model loss - all layers trained')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_loss_plot_all_"+str(run)+".png")
plt.close()

# Plot mae
plt.plot(training_metrics['mean_absolute_error'])
plt.plot(training_metrics['val_mean_absolute_error'])
plt.title('Model mean absolute error - all layers trained')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_mae_plot_all_"+str(run)+".png")
plt.close()

# Plot training & validation accuracies
plt.plot(training_metrics['accuracy'])
plt.plot(training_metrics['val_accuracy'])
plt.title('Model accuracy - all layers trained')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_acc_plot_all_"+str(run)+".png")
plt.close()

# Plot training & validation categorical accuracies
plt.plot(training_metrics['categorical_accuracy'])
plt.plot(training_metrics['val_categorical_accuracy'])
plt.title('Model categorical accuracy - all layers trained')
plt.ylabel('Categorical Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_cat_acc_plot_all_"+str(run)+".png")
plt.close()

# plot confusion matrix
sn.heatmap(conf_matrix_df, annot=True, cmap="viridis")
plt.title("Confusion Matrix - all layers trained")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
plt.savefig(data_location+data_set+"/2D_Transfer_Run "+str(run)+"/"+data_set_1+"_conf_matrix_all_"+str(run)+".png")
plt.close()