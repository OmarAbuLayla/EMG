# encoding: utf-8
import os
import scipy.io as sio
import random
import numpy as np
import librosa
from scipy import signal
import torch
from itertools import combinations
from python_speech_features import mfcc
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from opencv_transforms import *

# ==============================
# FILTERING FUNCTION
# ==============================
def filter(raw_data):
    """
    raw_data: np.array of shape (1, timepoints, NC)
    Returns filtered EMG, or original if too short
    """
    fs = 1000
    b1, a1 = signal.iirnotch(50, 30, fs)
    b2, a2 = signal.iirnotch(150, 30, fs)
    b3, a3 = signal.iirnotch(250, 30, fs)
    b4, a4 = signal.iirnotch(350, 30, fs)
    b5, a5 = signal.butter(4, [10/(fs/2), 400/(fs/2)], 'bandpass')

    min_len_required = max(len(a1), len(b1), len(a2), len(b2),
                           len(a3), len(b3), len(a4), len(b4),
                           len(a5), len(b5)) * 3  # filtfilt padlen = 3*(max(len(a),len(b))-1)
    
    if raw_data.shape[1] < min_len_required:
        # Too short to filter safely
        return raw_data

    # Apply filters sequentially along axis=1
    x = signal.filtfilt(b1, a1, raw_data, axis=1)
    x = signal.filtfilt(b2, a2, x, axis=1)
    x = signal.filtfilt(b3, a3, x, axis=1)
    x = signal.filtfilt(b4, a4, x, axis=1)
    x = signal.filtfilt(b5, a5, x, axis=1)
    return x

# ==============================
# MFSC FEATURE EXTRACTION
# ==============================
def EMG_MFSC(x):
    """
    x: shape (1, timepoints, NC)
    Returns: normalized MFSC features, shape (samples, channels, 36, 36)
    """
    # Trim first 250 samples (as in original code)
    x = x[:, 250:, :]

    n_mels = 36
    sr = 1000
    channel_list = []

    for j in range(x.shape[-1]):  # loop over channels
        # Initialize MFSC array
        mfsc_x = np.zeros((x.shape[0], 36, n_mels))

        for i in range(x.shape[0]):  # loop over samples
            signal_i = x[i, :, j]

            # Ensure the signal is long enough for n_fft
            if len(signal_i) < 2:
                # Skip empty or too-short signals
                mfsc_x[i, :, :] = np.zeros((36, n_mels))
                continue

            # Use min(n_fft, signal length)
            n_fft = min(200, len(signal_i))
            hop_length = 50
            signal_i = np.asfortranarray(signal_i)

            # Compute Mel spectrogram safely
            try:
                tmp = librosa.feature.melspectrogram(
                    y=signal_i,
                    sr=sr,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length
                )
                tmp = librosa.power_to_db(tmp).T
                # Ensure shape matches (36, n_mels)
                if tmp.shape[0] < 36:
                    pad_width = 36 - tmp.shape[0]
                    tmp = np.pad(tmp, ((0, pad_width), (0, 0)), mode='constant')
                elif tmp.shape[0] > 36:
                    tmp = tmp[:36, :]
                mfsc_x[i, :, :] = tmp
            except:
                # If any error occurs, fill zeros
                mfsc_x[i, :, :] = np.zeros((36, n_mels))

        # Add channel dimension
        mfsc_x = np.expand_dims(mfsc_x, axis=-1)
        channel_list.append(mfsc_x)

    # Concatenate channels
    data_x = np.concatenate(channel_list, axis=-1)

    # Normalize
    mu = np.mean(data_x)
    std = np.std(data_x) + 1e-8
    data_x = (data_x - mu) / std

    # Rearrange axes to (samples, channels, 36, 36)
    data_x = data_x.transpose(0, 3, 1, 2)

    return data_x

# ==============================
# DATASET CLASS
# ==============================
class MyDataset():
    def build_file_list(self, set, dir):
        trnList = []
        valList = []
        tstList = []

        subject_dir = os.path.join(dir, set.capitalize(), "EMG")
        emg_subject_list = sorted([
            os.path.join(subject_dir, s) 
            for s in os.listdir(subject_dir) 
            if os.path.isdir(os.path.join(subject_dir, s))
        ])

        n_subjects = len(emg_subject_list)
        n_train = int(n_subjects * 0.7)
        n_val = int(n_subjects * 0.1)
        n_test = n_subjects - n_train - n_val

        train_subjects = emg_subject_list[:n_train]
        val_subjects = emg_subject_list[n_train:n_train+n_val]
        test_subjects = emg_subject_list[n_train+n_val:]

        if set == 'train':
            target_subjects = train_subjects
        elif set == 'val':
            target_subjects = val_subjects
        elif set == 'test':
            target_subjects = test_subjects
        else:
            raise ValueError(f"Unknown set: {set}")

        for emg_dataset in target_subjects:
            sessions = os.listdir(emg_dataset)
            for session in sessions:
                session_path = os.path.join(emg_dataset, session)
                samples = os.listdir(session_path)
                for sample in samples:
                    label = sample.split('.')[0]
                    emgpath = os.path.join(session_path, sample)
                    entry = (label, emgpath)
                    if set == 'train':
                        trnList.append(entry)
                    elif set == 'val':
                        valList.append(entry)
                    elif set == 'test':
                        tstList.append(entry)

        random.shuffle(trnList)
        random.shuffle(valList)
        random.shuffle(tstList)

        if set == 'train':
            return trnList
        elif set == 'val':
            return valList
        elif set == 'test':
            return tstList

    def __init__(self, set, directory):
        self.set = set
        self.file_list = self.build_file_list(set, directory)
        print('Total num of samples: ', len(self.file_list))

    def __getitem__(self, idx):
        emg = sio.loadmat(self.file_list[idx][1])
        emg = np.expand_dims(emg["data"], axis=0)
        emg = filter(emg)
        emg = EMG_MFSC(emg)
        label = int(self.file_list[idx][0])
        emg = torch.FloatTensor(emg)
        return emg, label

    def __len__(self):
        return len(self.file_list)
