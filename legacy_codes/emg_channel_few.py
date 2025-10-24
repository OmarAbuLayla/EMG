# -*- coding: utf-8 -*-
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
N_KEEP = 8  # Number of virtual channels to keep

# =============================
# FUNCTION TO CREATE VIRTUAL CHANNELS
# =============================
def create_virtual_channels(emg):
    """
    Create virtual EMG channels:
    1. Compute all pairwise virtual channels: max(channel_i, channel_j) - min(channel_i, channel_j)
    2. Normalize all virtual channels (z-score)
    3. Keep the top N_KEEP channels based on variance
    """
    NC = emg.shape[2]
    virtual_list = []

    # Step 1: create all virtual channels
    for i, j in combinations(range(NC), 2):
        virtual_ch = np.maximum(emg[:, :, i], emg[:, :, j]) - np.minimum(emg[:, :, i], emg[:, :, j])
        virtual_list.append(virtual_ch)

    emg_virtual = np.stack(virtual_list, axis=2)

    # Step 2: normalize all virtual channels
    for ch in range(emg_virtual.shape[2]):
        ch_data = emg_virtual[:, :, ch]
        mean = np.mean(ch_data)
        std = np.std(ch_data)
        if std != 0:
            emg_virtual[:, :, ch] = (ch_data - mean) / std
        else:
            emg_virtual[:, :, ch] = ch_data - mean

    # Step 3: keep top N_KEEP channels by variance
    variances = [np.var(emg_virtual[:, :, ch]) for ch in range(emg_virtual.shape[2])]
    top_indices = np.argsort(variances)[-N_KEEP:]  # indices of top 8 channels
    emg_virtual = emg_virtual[:, :, top_indices]

    return emg_virtual

# =============================
# PROCESS ALL DATASETS
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

                # Save the treated data
                treated_file_path = os.path.join(treated_root, dataset, "EMG", subj, sess, file)
                os.makedirs(os.path.dirname(treated_file_path), exist_ok=True)
                sio.savemat(treated_file_path, {"data": emg_virtual})
                print(f"[DEBUG] Saved treated file (8 channels): {treated_file_path}")

print("\nAll sets processed! Virtual channels created and saved in:", treated_root)
