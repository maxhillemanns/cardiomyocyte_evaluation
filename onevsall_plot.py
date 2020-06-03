
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

data_location = "../results/onevsall/"

task_flag = 0

if task_flag == 0: 
    data = "sarc_data/"
    data_1 = "sarc"
    labels = ["null","1","2","3","4"]
elif task_flag == 1:
    data = "dir_data/"
    data_1 = "dir"
    labels = ["null","1","2","3","4"]
elif task_flag == 2:
    data = "cell_diff/"
    data_1 = "cell_diff"
    labels = ["null","adult", "cor4u", "iPSC", "neonatal", "w4ESC"]

conf_matrix = np.load(data_location+data+data_1+"_conf_matrix.npy")

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

# plot confusion matrix
sn.heatmap(conf_matrix_df, annot=True, cmap="viridis")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
plt.savefig(data_location+data+data_1+"_conf_matrix.png")
plt.close()


