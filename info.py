# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:46:57 2020

@author: mhill
"""

from datetime import datetime

def print_info(task_flag, run, status_flag):
    
    if task_flag == 0:
        task = "Sarcomerization Classification"
    elif task_flag == 1:
        task = "Directionality Classification"
    elif task_flag == 2:
        task = "Cell History Classification"
        
    log_time = datetime.now()
    date_time = log_time.strftime("%d/%m/%Y, %H:%M:%S")
    
    print("---------------------------\n")    
    print("+++ Experiment Info +++\n")
    print("Task: "+task+"\n")
    print("Run: "+str(run)+"\n")
    print("\n")
    print("+++ Experiment Log +++\n")
    
    if status_flag == 0:
        print("Experiment started - "+date_time+"\n")
    elif status_flag == 1:
        print("Experiment started\n")
        print("Data loaded - "+date_time+"\n")
    elif status_flag == 2:
        print("Experiment started\n")
        print("Data loaded\n")
        print("Data processed - "+date_time+"\n")
    elif status_flag == 3:
        print("Experiment started\n")
        print("Data loaded\n")
        print("Data processed\n")
        print("Data augmentated - "+date_time+"\n")
    elif status_flag == 4:
        print("Experiment started\n")
        print("Data loaded\n")
        print("Data processed\n")
        print("Data augmentated\n")
        print("Data split in sets - "+date_time+"\n")
    elif status_flag == 5:
        print("Experiment started\n")
        print("Data loaded\n")
        print("Data processed\n")
        print("Data augmentated\n")
        print("Data split in sets\n")
        print("Model built and compiled - "+date_time+"\n")
    elif status_flag == 6:
        print("Experiment started\n")
        print("Data loaded\n")
        print("Data processed\n")
        print("Data augmentated\n")
        print("Data split in sets\n")
        print("Model built and compiled\n")
        print("Model trained - "+date_time+"\n")
    elif status_flag == 7:
        print("Experiment started\n")
        print("Data loaded\n")
        print("Data processed\n")
        print("Data augmentated\n")
        print("Data split in sets\n")
        print("Model built and compiled\n")
        print("Model trained\n")
        print("Model saved - "+date_time+"\n")
    elif status_flag == 8:
        print("Experiment started\n")
        print("Data loaded\n")
        print("Data processed\n")
        print("Data augmentated\n")
        print("Data split in sets\n")
        print("Model built and compiled\n")
        print("Model trained\n")
        print("Model saved\n")
        print("Model evaluated - "+date_time+"\n")
    elif status_flag == 9:
        print("Experiment started\n")
        print("Data loaded\n")
        print("Data processed\n")
        print("Data augmentated\n")
        print("Data split in sets\n")
        print("Model built and compiled\n")
        print("Model trained\n")
        print("Model saved\n")
        print("Model evaluated\n")
        print("Experiment done - "+date_time+"\n")
    