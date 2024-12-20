"""
Filename:        MNISTandFMNIST_ReadData.py
Author:          Jia-Yi LI
Last Modified:   2024-12-20
Version:         1.0
Description:     Read experimental data of MNIST and Fashion-MNIST from physical reservoir devices.
License:         MIT License
                 Copyright (c) 2024 Nanyang Technological University
"""

import numpy as np
import os
from tqdm import tqdm
import xlrd
import pandas as pd

# Define directories for data and labels
root_drc = os.path.abspath('DIRC_TO_DATA')
directory_data = os.path.join(root_drc, 'Data')
directory_label = os.path.join(root_drc, 'Label')

# Function to read and process Excel data
def readExcelData():
    """Read IMeasCh2 column from Excel files in the data directory."""
    dfs = []
    tab_names = ['Data'] + [f'Cycle{i}' for i in range(2, 11)]

    # Rename and pad files if necessary
    from Data.usrUtilis import pad_and_rename_files
    pad_and_rename_files(directory_data)

    # Read each file and tab
    for filename in tqdm(sorted(os.listdir(directory_data))):
        if filename.endswith(".xls"):
            wb = xlrd.open_workbook(os.path.join(directory_data, filename), logfile=open(os.devnull, 'w'))
            xls = pd.ExcelFile(wb, engine='xlrd')
            for tab_name in tab_names:
                df = pd.read_excel(xls, tab_name, usecols=['IMeasCh2'])
                dfs.append(df)
    t_pd = pd.read_excel(xls, tab_name, usecols=['TimeOutput'])
    return dfs, t_pd

# Process the current data
from Data.rd_PRC import rd_data_waveform
I_rd, t, V_rd = rd_data_waveform(root_drc, force_read=False)
I_agg_all = I_rd
V_rd = V_rd[0]
t = t.flatten()

# Average the data across cycles
I_avg_all = np.mean(I_agg_all, axis=1)
v_avg_all = np.mean(V_rd, axis=1)

print(f"Processed Data Shapes - I_avg_all: {I_avg_all.shape}, t: {t.shape}, v_avg_all: {v_avg_all.shape}")

# Reshape the data for reservoir computing
numofpoints = 12
read_start = 53
read_interval = 88
numofrows = 14

I_read_14pts = np.empty((16, I_agg_all.shape[0], I_agg_all.shape[1], numofpoints))
for i in range(16):
    start = read_start + read_interval * i
    end = read_start + numofpoints + read_interval * i
    I_read_14pts[i] = I_agg_all[:, :, start:end]

I_read_14pts_stack = I_read_14pts.reshape((I_read_14pts.shape[0], I_read_14pts.shape[1], -1))
I_read_14pts_stack_neg = np.negative(I_read_14pts_stack)
rc_read_14pts = np.reshape(I_read_14pts_stack_neg, (I_read_14pts_stack_neg.shape[0], -1, numofrows, I_read_14pts_stack_neg.shape[2]))
rc_read_14pts = np.transpose(rc_read_14pts, (1, 0, 2, 3))

print(f"Reservoir Computing Data Shapes - rc_read_14pts: {rc_read_14pts.shape}")

# Calculate energy consumption
e_all = np.multiply(I_rd, V_rd) * 5  # ns per point
e_ind = np.reshape(e_all, (rc_read_14pts.shape[0], rc_read_14pts.shape[2], I_rd.shape[1], I_rd.shape[2]))

# Compute total energy per device
img_num, dvc_chan, cycle_num = e_ind.shape[:3]
total_e_ind = np.empty((img_num, dvc_chan, cycle_num))
for img_idx in range(img_num):
    for dvc_idx in range(dvc_chan):
        for cyc in range(cycle_num):
            total_e_ind[img_idx, dvc_idx, cyc] = np.trapz(e_ind[img_idx, dvc_idx, cyc], t)

total_e_ind = np.abs(total_e_ind)
img_e = np.sum(total_e_ind, axis=1)
print(f"Total Energy Data Shapes - total_e_ind: {total_e_ind.shape}, img_e: {img_e.shape}")

# Compute average and median energy per device
avg_energy = np.average(total_e_ind) * 1e12
med_energy = np.median(total_e_ind) * 1e12
print(f"Average Energy per Device: {avg_energy:.2f} pJ")
print(f"Median Energy per Device: {med_energy:.2f} pJ")
