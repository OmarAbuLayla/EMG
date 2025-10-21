# emg_channel.py
# -*- coding: utf-8 -*-


# channeling.py
# -----------------------------
# This script creates virtual EMG channels for all samples in Train, Val, and Test sets.
# The virtual channels are computed as max(channel_i, channel_j) - min(channel_i, channel_j) 
# for every pair of original channels. The new files are saved in the same folder structure
# but under a new root folder (to avoid overwriting the original dataset).
# -----------------------------




import os
import scipy.io as sio
import numpy as np
from itertools import combinations

# =============================
# CONFIGURATION
# =============================
original_root = r"C:\Users\ompis\Desktop\work GJU\Codes\AVE-Speech"
treated_root  = r"D:\Omar\AVE-Speech_treated_few_channels"

sets = ["Train", "Val", "Test"]

# =============================
# FUNCTION TO CREATE VIRTUAL CHANNELS
# =============================
# =============================
# FUNCTION TO CREATE VIRTUAL CHANNELS AND NORMALIZE
# =============================
def create_virtual_channels(emg):
    """
    emg: np.array with shape (1, timepoints, NC)
    NC = number of original EMG channels
    Returns:
        emg_virtual: np.array with shape (1, timepoints, NC*(NC-1)/2), normalized per channel
    """
    NC = emg.shape[2]  # Number of original EMG channels
    virtual_list = []

    for i, j in combinations(range(NC), 2):
        virtual_ch = np.maximum(emg[:, :, i], emg[:, :, j]) - np.minimum(emg[:, :, i], emg[:, :, j])
        virtual_list.append(virtual_ch)

    emg_virtual = np.stack(virtual_list, axis=2)

    # ----------------------------
    # Normalize each virtual channel (z-score)
    # ----------------------------
    for ch in range(emg_virtual.shape[2]):
        ch_data = emg_virtual[:, :, ch]
        mean = np.mean(ch_data)
        std = np.std(ch_data)
        if std != 0:
            emg_virtual[:, :, ch] = (ch_data - mean) / std
        else:
            emg_virtual[:, :, ch] = ch_data - mean  # avoid division by zero

    return emg_virtual


# =============================
# PROCESS ALL DATA
# =============================
for dataset in sets:
    print(f"\nProcessing {dataset} set...")

    original_set_path = os.path.join(original_root, dataset, "EMG")
    if not os.path.exists(original_set_path):
        print(f"[WARNING] Original path does not exist: {original_set_path}")
        continue

    subjects = sorted([s for s in os.listdir(original_set_path) if os.path.isdir(os.path.join(original_set_path, s))])

    for subj in subjects:
        sessions = sorted([sess for sess in os.listdir(os.path.join(original_set_path, subj)) 
                           if os.path.isdir(os.path.join(original_set_path, subj, sess))])
        for sess in sessions:
            files = [f for f in os.listdir(os.path.join(original_set_path, subj, sess)) if f.endswith(".mat")]
            for file in files:
                # Load original .mat file
                original_file_path = os.path.join(original_set_path, subj, sess, file)
                mat = sio.loadmat(original_file_path)
                emg = mat["data"]

                # Ensure shape is (1, timepoints, NC)
                if emg.ndim == 2:
                    emg = np.expand_dims(emg, axis=0)
                elif emg.ndim != 3:
                    raise ValueError(f"Unexpected EMG shape in {original_file_path}: {emg.shape}")

                # Create virtual channels
                emg_virtual = create_virtual_channels(emg)

                # Build treated save path
                treated_file_path = os.path.join(treated_root, dataset, "EMG", subj, sess, file)
                os.makedirs(os.path.dirname(treated_file_path), exist_ok=True)

                # Save treated .mat
                sio.savemat(treated_file_path, {"data": emg_virtual})
                print(f"[DEBUG] Saved treated file: {treated_file_path}")

print("\nAll sets processed! Virtual channels created and saved in:", treated_root)
