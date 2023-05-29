

import pandas as pd
import datetime
import time
from csv import reader
import matplotlib.pyplot as plt
from pydub import AudioSegment
import wave
from scipy.io import wavfile
from scipy.io.wavfile import write
import numpy as np
import os
import contextlib
import torchaudio
import csv

#to do: check no else wav is used since now we only have one wav per conv

def convert_to_wav(path, extention = 'm4a'):
    l_direct = os.listdir(path)
    for direct in l_direct:
        #os.mkdir(path_wav+direct)
        l_files = os.listdir(path+direct)
        for file in l_files:
            if extention in file:
                if extention == 'm4a':
                    if len(file) == 15: #if there is also one wav for both speakers, keep only the ones were each speaker is recodred speratly
                        track = AudioSegment.from_file(path+'/'+direct+'/'+file,  format=extention)
                        file_handle = track.export(path+direct+'/'+file[:-3]+'wav', format='wav')
                if extention == 'mp4':
                    track = AudioSegment.from_file(path+'/'+direct+'/'+file,  format=extention)
                    file_handle = track.export(path+direct+'/'+file[:-3]+'wav', format='wav')

def create_empty_csv(csv_path,csv_name):
    column_names = ["filename","label","nbframes","folder"] 

    with open(csv_path+csv_name, 'w') as f:
    # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(column_names)

def return_sec(time_string):    
    x = time.strptime(time_string.split('.')[0],'%H:%M:%S')
    s = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    ms = float('0.'+time_string[-3:])
    return s+ms

# open file in read mode
def extract_df_times_csv_file(data_csv,samplerate,A):
    l_feedback = []
    l_resp = []
    with open(data_csv, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            #if x in row:
            if 'SpeechFunction' in row:
                if row[8][:2] == A:
                    if row[8][-8:] == "Feedback":
                        l_feedback.append([return_sec(row[2])*samplerate,return_sec(row[4])*samplerate,return_sec(row[6])*samplerate,row[8]])
                    if row[8][-8:] == "Response":
                        l_resp.append([return_sec(row[2])*samplerate,return_sec(row[4])*samplerate,return_sec(row[6])*samplerate,row[8]])
    df_r = pd.DataFrame(l_resp) 
    df_r.to_csv('resp.csv') 
    df_f = pd.DataFrame(l_feedback) 
    df_f.to_csv('feedback.csv') 
    if l_feedback != []:     
        df_f.columns =['onset', 'end', 'duration', 'label']
    if l_resp != []:
        df_r.columns =['onset', 'end', 'duration', 'label']
    return df_f,df_r
        
def cut_one_wav(df,wav,backchannel_type,direct,folder,direct_cut,nb_frames_before_onset,nb_frames_after_offset,speaker):
    samplerate, data = wavfile.read(direct+folder+wav)
    s = len(df)
    for i in range (s):
        onset = int(get_onset(df,i))-nb_frames_before_onset
        offset = int(get_end(df,i))+nb_frames_after_offset
        if onset+100<offset:
            newAudio = data[onset:offset]
            write(direct_cut+'/'+folder+'/'+wav[:-4]+'_'+str(i)+'_'+backchannel_type+speaker+".wav", samplerate, newAudio.astype(np.int16))
    
def cut_wav_file(data_csv,wav1,wav2,direct,folder,direct_cut,nb_frames_before_onset,nb_frames_after_offset):
    samplerate, data = wavfile.read(direct+folder+wav1)
    df1_f,df1_r = extract_df_times_csv_file(data_csv,samplerate,'A1')
    df2_f,df2_r = extract_df_times_csv_file(data_csv,samplerate,'A2')
    
    cut_one_wav(df1_f,wav1,"feedback",direct,folder,direct_cut,nb_frames_before_onset,nb_frames_after_offset,"A1")
    cut_one_wav(df2_f,wav2,"feedback",direct,folder,direct_cut,nb_frames_before_onset,nb_frames_after_offset,"A2")
    cut_one_wav(df1_r,wav1,"response",direct,folder,direct_cut,nb_frames_before_onset,nb_frames_after_offset,"A1")
    cut_one_wav(df2_r,wav2,"response",direct,folder,direct_cut,nb_frames_before_onset,nb_frames_after_offset,"A2")
    
       
def get_onset(df,i):
    return df.at[i, 'onset']
    
def get_end(df,i):
    return df.at[i, 'end']


def search_extention_in_given_folder (folder,path,extention,size = None):
    l_files = os.listdir(path+folder)
    l = []
    
    for file in l_files:
        if file.endswith(extention):
            if size == None:
                l.append(file)
            else:
                if len(file) == size:
                    l.append(file)
    return l

def cut_all_wav(path_rec,path_annot,path_cut,nb_frames_before_onset,nb_frames_after_offset = 0,from_mp4_file = False):
    l_folder = os.listdir(path_rec) 
    for folder in l_folder:
        if not folder == "BC":
            os.mkdir(path_cut+folder)
            folder = folder+'/'
            if not from_mp4_file:
                [wav1,wav2] = search_extention_in_given_folder (folder,path_rec,".wav",15)
                adult_1 = wav1[3:5]
                wav1, wav2 = [wav1, wav2] if wav1[-6:-4] == adult_1 else [wav2, wav1]
            if from_mp4_file:
                l = search_extention_in_given_folder (folder,path_rec,".wav",12)
                wav1 = l[0]
                wav2= l[0]
            [csv_file] = search_extention_in_given_folder (folder,path_annot,".csv")
            data_csv = path_annot+folder+csv_file
            cut_wav_file(data_csv,wav1,wav2,path_rec,folder,path_cut,nb_frames_before_onset,nb_frames_after_offset)
            
            

def create_csv_from_cut_wav(path_cut,nb_frames_before_onset,nb_frames_after_offset):
    #print("TO DO: verify if max supposed to be is max frames or max duration")
    l_folders = os.listdir(path_cut)
    l_csv = []
    for folder in l_folders:
        l_wav = os.listdir(path_cut+folder)
        
        for elem in l_wav:
            
            if "response" in elem:
                label = 0
            else :
                label = 1
                
            #with contextlib.closing(wave.open(path_cut+'/'+folder+'/'+elem,'r')) as f:
            
            soundData, sample_rate = torchaudio.load(path_cut+'/'+folder+'/'+elem,normalize=True)
            mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)(soundData)
            mfcc_size = mfcc.shape[-1]
            frames = soundData.shape[-1]
            #frames = f.getnframes()+nb_frames_before_onset+nb_frames_after_offset
            if frames > 1000:
                l_csv.append([elem,label,frames+nb_frames_before_onset+nb_frames_after_offset,folder,mfcc_size])

    df = pd.DataFrame(l_csv) 
    df.columns =['filename', 'label', 'nbframes','folder','mfcc_size']
    df.to_csv(path_cut+'filenames_labels_nbframes2.csv') 
    return df
        




path_rec = "../data/Adult-rec/"
path_annot = "../data/Annotations-adults/"
path_cut = "../data/cut_wav_with_data_before_onset_without_BC/"
path_cut = "../data/cut_wav_with_data_before_onset_without_BC_from_mp4/"
nb_frames_before_onset = 50000
nb_frames_after_offset = 5000

#convert_to_wav(path_rec,'mp4')

#cut_all_wav(path_rec,path_annot,path_cut,nb_frames_before_onset,nb_frames_after_offset,True)
df = create_csv_from_cut_wav(path_cut,nb_frames_before_onset,nb_frames_after_offset)

