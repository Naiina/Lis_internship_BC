import os
import pandas as pd
import matplotlib.pyplot as plt

import statistics

def get_chance_proba(idx_folder,l_folder,csv,csv_path):
    nb_ones = 0
    nb_zeros = 0
    folder = l_folder[idx_folder]
    df = pd.read_csv(csv_path+csv)
    dff = df[df["folder"]==folder]
    print(dff)
    df1 = dff[df["label"] == 1]
    
    df0 = dff[df["label"] == 0]
    
    return max(len(df1),len(df0))/len(dff)



def plot_fig(idx_test_set):
    
    

    df1 = df[df["ixd_test_set"] == idx_test_set]

    l_loss = list(df1["loss"])
    l_accuracy =  list(df1["correct"])
    ii = get_chance_proba(idx_test_set,l_folders,csv_filename,path_cut_before )
    print(ii)

    plt.plot(l_loss, label = "loss")
    plt.plot(l_accuracy, label = "accuracy (%)")
    plt.xlabel('nb epochs')

    #plt.hlines(ii,0,30, colors='k', linestyles='dashed', label = "chance level")
    plt.legend()
    plt.title("lstm 1 layer, 128 hidden, lr = 0.001, weighted loss")
    plt.savefig("../results/weighted_cross_val_001_bis"+str(idx_test_set))
    plt.show()


path_cut_before = "../data/cut_wav_with_data_before_onset_without_BC_from_mp4/"
csv_filename = "filenames_labels_nbframes.csv"
l_folders = [elem for elem in os.listdir(path_cut_before) if os.path.isdir(path_cut_before+elem)]
idx_test_set = 5
df = pd.read_csv("../results/cross_val_128_1_lr_0_001_weighted_bis.csv")

def stats():
    dfff = df[df["epoch"] == 30]
    l_accu = list(dfff["correct"])
    
    l_chance = []
    for idx in range(9):
        l_chance.append(get_chance_proba(idx,l_folders,csv_filename,path_cut_before))
    print(l_accu)
    print(l_chance)
    print(statistics.mean(l_accu))
    print(statistics.mean(l_chance))

plot_fig(1)
#stats()