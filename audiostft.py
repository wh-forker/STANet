import torch
import numpy as np
from scipy.io import wavfile
import os
import h5py
from PIL import Image
from scipy import signal
import cv2
import random
import soundfile as sf
import resampy
import numpy as np
import json
import argparse
import math
import os
from torchvision import datasets, transforms

video_path = "./Img2/"
audio_dir = "./Audio2/"
feature_path = "./Audio3/"
ori_name = os.listdir(video_path)
for file in range(0, len(ori_name)):
    print(ori_name[file])
    if not os.path.exists(os.path.join(feature_path, ori_name[file])):
        os.makedirs(os.path.join(feature_path, ori_name[file]))
    ficpath = os.path.join(video_path, ori_name[file])
    ficname = os.listdir(ficpath)
    for fs in range(0, len(ficname)):
        save_path = os.path.join(feature_path, ori_name[file], ficname[fs])
        print(ficname[fs])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        picpath = os.path.join(ficpath, ficname[fs])
        picname = os.listdir(picpath)
        newlist = []
        for names in picname:
            if names.endswith(".jpg"):
                newlist.append(names)

        audio_path = os.path.join(
            audio_dir, ori_name[file], ficname[fs], ficname[fs] + '.wav')
        
        if os.path.exists(audio_path):

            samples, samplerate = sf.read(audio_path)

            if len(samples.shape) > 1:
                samples = np.mean(samples, axis=1)

            SAMPLE_RATE = 16000
            if samplerate != SAMPLE_RATE:
                samples = resampy.resample(samples, samplerate, SAMPLE_RATE)  # 采样速率转换44100->16000
            T = len(newlist)
            long = 8000
            L = samples.shape[0]
            # IoU = math.floor(long-(L-long-(long/500))/(T-1))
            IoU = math.ceil(long-(L-long)/(T-1))
            spectrogramall = np.zeros([T, 1, 257, 48])
            for picp in range(0, len(newlist)):
                s = picp*(long-IoU)
                e = picp*(long-IoU)+long
                resamples = samples[s:e]
                resamples[resamples > 1.] = 1.
                resamples[resamples < -1.] = -1.
                _, _, spectrogram = signal.spectrogram(resamples, SAMPLE_RATE, nperseg=512, noverlap=353)
                spectrogram = np.log(spectrogram + 1e-7)
                mean = np.mean(spectrogram)
                std = np.std(spectrogram)
                audio_output = np.divide(spectrogram-mean, std+1e-9)  # 257，61
                print(newlist[picp])
                with h5py.File(save_path+'\\'+newlist[picp][:-4]+'_asp.h5', 'w') as hf:
                    hf.create_dataset("dataset",  data=audio_output)

                    # audio_output = audio_input.float()
                    
