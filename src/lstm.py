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





class UrbanSoundDataset(Dataset):
    # Wrapper for the UrbanSound8K dataset
    # Argument List
    # path to the UrbanSound8K csv file
    # path to the UrbanSound8K audio files
    # list of folders to use in the dataset

    
    def __init__(self, csv_filename, path_cut,folderList):
        csvData = pd.read_csv(path_cut+csv_filename)
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csvData)): 
            if csvData.iloc[i, 4] in folderList:
                self.file_names.append(csvData.iloc[i, 1])
                self.labels.append(csvData.iloc[i, 2])
                self.folders.append(csvData.iloc[i, 4])
        self.path_cut = path_cut
        self.max = csvData['mfcc_size'].max()
        self.folderList = folderList
        
        

    def __getitem__(self, index):
        # format the file path and load the file
        #max_frames = int(self.max/150)
        path = self.path_cut + str(self.folders[index]) + "/" +self.file_names[index]
        soundData, sample_rate = torchaudio.load(path)
        #print(path,soundData.shape)
        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)(soundData)
        mfcc_temporal_size = mfcc.shape[2]
        #soundData = torch.mean(sound, dim=0, keepdim=True)
        padded_mfcc = torch.zeros([1,40, self.max])  # tempData accounts for audio clips that are too short
        
       
        padded_mfcc[:,:, :mfcc_temporal_size] = mfcc
        

        #mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(soundData)  # (channel, n_mels, time)
        #mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
        #mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)(soundData)  # (channel, n_mfcc, time)
        #mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        # spectogram = torchaudio.transforms.Spectrogram(sample_rate=sample_rate)(soundData)
        #feature = torch.cat([mel_specgram, mfcc], axis=1)
        #feature = mfcc
        
        return padded_mfcc[0].permute(1, 0), self.labels[index], mfcc_temporal_size
    def __len__(self):
        return len(self.file_names)
    
    





class AudioLSTM(nn.Module):

    def __init__(self, n_feature=5, out_feature=5, n_hidden=256, n_layers=2, drop_prob=0.1):
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
    
        
def train(model, epoch):
    l_loss = []
    model.train()
    for batch_idx, (data, target, seq_len) in enumerate(train_loader):
        #print(target)
        data = data.to(device)
        target = target.to(device)

        model.zero_grad()
        output, hidden_state = model(data, seq_len,model.init_hidden(hyperparameters["batch_size"]))
        
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        #if batch_idx % log_interval == 0: #print training stats
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss))
        l_loss.append(loss.data.item())
    return l_loss
            
            
def test(model, epoch):
    model.eval()
    correct = 0
    y_pred, y_target = [], []
    for data, target, seq_len in test_loader:
        data = data.to(device)
        target = target.to(device)
        
        output, hidden_state = model(data, seq_len, model.init_hidden(hyperparameters["batch_size"]))
        
        pred = torch.max(output, dim=1).indices
        correct += pred.eq(target).cpu().sum().item()
        y_pred = y_pred + pred.tolist()
        y_target = y_target + target.tolist()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


device = torch.device("cpu")
print(device)

#csv_path = '/kaggle/input/urbansound8k/UrbanSound8K.csv'
#file_path = '/kaggle/input/urbansound8k/'

path_cut_before = "../data/cut_wav_with_data_before_onset_without_BC_from_mp4/"
csv_filename = "filenames_labels_nbframes.csv"
l_folders = [elem for elem in os.listdir(path_cut_before) if os.path.isdir(path_cut_before+elem)]

train_set = UrbanSoundDataset(csv_filename, path_cut_before, l_folders[:-2] )
test_set = UrbanSoundDataset(csv_filename, path_cut_before,l_folders[-2])
validation_set = UrbanSoundDataset(csv_filename, path_cut_before,l_folders[-1])

#print("Train set size: " + str(len(train_set)))
#print("Test set size: " + str(len(test_set)))

#TO INVESTIGATE: No batch size > test_set/2, because drop last batch is smaller the the others 
batch_size = len(train_set)//2 
hyperparameters = {"lr": 0.001, "weight_decay": 0.0001, "batch_size": batch_size, "in_feature": 40, "out_feature": 2}
#hyperparameters = {"lr": 0.01, "weight_decay": 0.0001, "batch_size": 20, "in_feature": 40, "out_feature": 2}



kwargs = {}  # needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)

model = AudioLSTM(n_feature=hyperparameters["in_feature"], out_feature=hyperparameters["out_feature"])
model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
criterion = nn.CrossEntropyLoss()
clip = 5  # gradient clipping

log_interval = 10
l_l_loss = []
for epoch in range(1, 3):
    # scheduler.step()
    l_loss = train(model, epoch)
    l_l_loss.append(l_loss)
    test(model, epoch)
    

l_l_loss.to_csv(path_cut_before,)



with open('GFG2.csv', 'w') as f:
    i = 0
    # using csv.writer method from CSV package
    write = csv.writer(f)
    
    for k in hyperparameters.keys():
        write.writerows([k,str(hyperparameters[k])])
    for elem in l_l_loss:
        i+=1
        write.writerow("epoch",i,elem)
    
