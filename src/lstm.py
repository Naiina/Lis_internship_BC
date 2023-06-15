# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:42:31 2023

@author: Lenovo
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
from scipy.io import wavfile
import csv
import random

import matplotlib.pyplot as plt

run_lstm = True



def get_pro(l_folder,csv,csv_path):
    nb_ones = 0
    nb_zeros = 0
    total = 0
    for folder in l_folder:
        df = pd.read_csv(csv_path+csv)
        dff = df[df["folder"]==folder]
        df1 = dff[df["label"] == 1]
        df0 = dff[df["label"] == 0]
        nb_zeros += len(df0)
        nb_ones += len(df1)
    total = nb_ones + nb_zeros
    return nb_zeros/total, nb_ones/total


class UrbanSoundDataset(Dataset):
    # Wrapper for the UrbanSound8K dataset
    # Argument List
    # path to the UrbanSound8K csv file
    # path to the UrbanSound8K audio files
    # list of folders to use in the dataset

    
    def __init__(self, csv_filename, path_cut,folderList):
        csvData = pd.read_csv(path_cut+csv_filename)
        
        self.file_names = []
        self.labels = []
        self.folders = []
        
        for i in range(0, len(csvData)): 
            if csvData.iloc[i, 4] in folderList:
                self.file_names.append(csvData.iloc[i, 1])
                self.labels.append(csvData.iloc[i, 2])
                self.folders.append(csvData.iloc[i, 4])
        self.path_cut = path_cut
        self.max = csvData['mfcc_size'].max()
        self.folderList = folderList
        
        

    def __getitem__(self, index):
       
        path = self.path_cut + str(self.folders[index]) + "/" +self.file_names[index]
        soundData, sample_rate = torchaudio.load(path)
      
        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)(soundData)
        mfcc_temporal_size = mfcc.shape[2]
       
        padded_mfcc = torch.zeros([1,40, self.max])  # tempData accounts for audio clips that are too short
        
       
        padded_mfcc[:,:, :mfcc_temporal_size] = mfcc
        

        
        
        return padded_mfcc[0].permute(1, 0), self.labels[index], mfcc_temporal_size
    def __len__(self):
        return len(self.file_names)
    
    





class AudioLSTM(nn.Module):

    def __init__(self, n_feature=5, out_feature=5, n_hidden=128, n_layers=1, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature

        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, out_feature)

    def forward(self, x, seq_len, hidden):
        # x.shape (batch, seq_len, n_features)
        l_out, l_hidden = self.lstm(x, hidden)
        

        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(l_out)
        
        #print(seq_len)
        #print (out[:, -1, :].shape)
        #print (out[range(x.shape[0]), seq_len-1].shape)
        
        # out.shape (batch, out_feature)
        #out = self.fc(out[:, -1, :])
        
        #new:
        last_hidden_states = out[range(x.shape[0]), seq_len-1]
        out = self.fc(last_hidden_states)

        # return the final output and the hidden state
        return out, l_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden
    
        
def train(model, epoch, k):
    l_loss = []
    model.train()
    for batch_idx, (data, target, seq_len) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        model.zero_grad()
        output, hidden_state = model(data, seq_len,model.init_hidden(hyperparameters["batch_size"]))
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        #if batch_idx % log_interval == 0: #print training stats
        print("test_set nb ",k)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss))
        l_loss.append(loss.data.item())
    return l_loss
            
            
def test(model, epoch,test_loader):
    model.eval()
    correct = 0
    y_pred, y_target = [], []
    for data, target, seq_len in test_loader:
        data = data.to(device)
        target = target.to(device)
        
        output, hidden_state = model(data, seq_len, model.init_hidden(1))
        
        pred = torch.max(output, dim=1).indices
        correct += pred.eq(target).cpu().sum().item()
        y_pred = y_pred + pred.tolist()
        y_target = y_target + target.tolist()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct/len(test_loader.dataset)




#csv_path = '/kaggle/input/urbansound8k/UrbanSound8K.csv'
#file_path = '/kaggle/input/urbansound8k/'

path_cut_before = "../data/cut_wav_with_data_before_onset_without_BC_from_mp4/"
csv_filename = "filenames_labels_nbframes.csv"
l_folders = [elem for elem in os.listdir(path_cut_before) if os.path.isdir(path_cut_before+elem)]


#print("Train set size: " + str(len(train_set)))
#print("Test set size: " + str(len(test_set)))



if run_lstm:
    
    l_l_loss = []
    #l_l_loss_end = []
    l_correct = []
    #l_lr = [0.01,0.005,0.001]
    lr = 0.001
    l_idx = []
    l_epoch = []
    
    ll_lr = []
    device = torch.device("cpu")
    print(device)
    for idx_test_set in [1,2]:
        #idx_test_set = random.randint(0,8)
        l_train_fold = l_folders[:idx_test_set] +  l_folders[idx_test_set+1:]
        train_set = UrbanSoundDataset(csv_filename, path_cut_before, l_train_fold )
        test_set = UrbanSoundDataset(csv_filename, path_cut_before,l_folders[idx_test_set])
        #validation_set = UrbanSoundDataset(csv_filename, path_cut_before,l_folders[-1])
    
    #TO INVESTIGATE: No batch size > test_set/2, because drop last batch is smaller the the others 
        batch_size = len(train_set)//2 
        hyperparameters = { "weight_decay": 0.0001, "batch_size": batch_size, "in_feature": 40, "out_feature": 2}
        #hyperparameters = {"lr": 0.01, "weight_decay": 0.0001, "batch_size": 20, "in_feature": 40, "out_feature": 2}
        
        
        
        kwargs = {}  # needed for using datasets on gpu
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, drop_last=True, **kwargs)
        #validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=True, drop_last=True, **kwargs)
        
        
        model = AudioLSTM(n_feature=hyperparameters["in_feature"], out_feature=hyperparameters["out_feature"])
        model.to(device)
        print(model)
    
    

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=hyperparameters['weight_decay'])
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        prob_0, prob_1 = get_pro(l_train_fold,csv_filename,path_cut_before)
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1-prob_0, 1-prob_1]))
        clip = 5  # gradient clipping
        
        
        epoch = 0
        nnn = 0
        mem_acc = 0
        while nnn< 3 and epoch <60 :
            
            epoch = epoch+1
            
            #for epoch in range(1, 31):
            l_loss = train(model, epoch, idx_test_set)
            correct = test(model, epoch, test_loader)
            l_l_loss.append(l_loss[-1])
            l_correct.append(correct)
            l_idx.append(idx_test_set)
            l_epoch.append(epoch)
            
            if epoch > 15:
                if correct < mem_acc:
                    nnn = nnn+1
                else:
                    nnn = 0
                    mem_acc = correct
 
        #valid = test(model, epoch, validation_loader)
        #for epoch in range(1, 61):
        #    l_valid.append(valid)
    
        
    #d = {"epoch":l_epoch,"loss":l_l_loss,"correct":l_correct,"lr":ll_lr,"validation":l_valid}
    d = {"epoch":l_epoch,"loss":l_l_loss,"correct":l_correct,"ixd_test_set":l_idx}
    df = pd.DataFrame.from_dict(d)
    df.to_csv("../results/cross_val_128_1_lr_0_001_weighted_bis.csv")

