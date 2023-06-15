# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:34:48 2023

@author: Lenovo
"""
import pandas as pd
import matplotlib.pyplot as plt




df = pd.read_csv("256_1_lr.csv")


l_1 = list(df["loss_batch_end"])[:25]
l_2 = list(df["loss_batch_end"])[25:50]
l_3 = list(df["loss_batch_end"])[50:]
plt.plot(l_1, label = "lr = 0.01")
plt.plot(l_2, label = "lr = 0.005")
plt.plot(l_3, label = "lr = 0.001")
plt.legend()
plt.title("lstm 1 layer, 256 hidden")
plt.savefig("lstm 1 layer, 256 hidden")
plt.show()



df = pd.read_csv("128_lr.csv")


l_1 = list(df["loss_batch_end"])[:25]
l_2 = list(df["loss_batch_end"])[25:50]
l_3 = list(df["loss_batch_end"])[50:]
plt.plot(l_1, label = "lr = 0.01")
plt.plot(l_2, label = "lr = 0.005")
plt.plot(l_3, label = "lr = 0.001")
plt.legend()
plt.title("lstm 2 layers, 128 hidden")
plt.savefig("lstm 2 layers, 128 hidden")
plt.show()



df = pd.read_csv("256_lr.csv")


l_1 = list(df["loss_batch_end"])[:25]
l_2 = list(df["loss_batch_end"])[25:50]
l_3 = list(df["loss_batch_end"])[50:]
plt.plot(l_1, label = "lr = 0.01")
plt.plot(l_2, label = "lr = 0.005")
plt.plot(l_3, label = "lr = 0.001")
plt.legend()
plt.title("lstm 2 layers, 265 hidden")
plt.savefig("lstm 2 layers, 265 hidden")
plt.show()

#%%


df = pd.read_csv("128_1_lr.csv")


l_1 = list(df["loss_batch_end"])[:25]
l_2 = list(df["loss_batch_end"])[25:50]
l_3 = list(df["loss_batch_end"])[50:]
plt.plot(l_1, label = "lr = 0.01")
plt.plot(l_2, label = "lr = 0.005")
plt.plot(l_3, label = "lr = 0.001")
plt.legend()
plt.title("lstm 1 layer, 128 hidden")
plt.savefig("lstm 1 layer, 128 hidden")
plt.show()
