"""
Filename:        HAR.py
Author:          Jia-Yi LI
Last Modified:   2024-12-20
Version:         1.0
Description:     Read experimental data of MNIST and Fashion-MNIST from physical reservoir devices.
                 Copyright (c) 2024 Nanyang Technological University
"""
import numpy as np
import os
import pandas as pd
from Data.dataset.HAR import readWeizmannRevisited as HAR
from tqdm import tqdm
import Data.usrUtilis as usrUtilis
import Data.rd_PRC as rdPRC

# Define directories
root_drc = os.path.abspath('DIRC_TO_DATA_FILES')
directory_data = os.path.join(root_drc, 'Data')

force_read = True

def readExcelData():
    """
    Reads and processes Excel files to extract data from specific tabs.
    """
    dfs = []  # List to hold all DataFrames
    tab_names = ['Data'] + [f'Cycle{i}' for i in range(2, 11)]

    usrUtilis.pad_and_rename_files(directory_data)

    for filename in tqdm(sorted(os.listdir(directory_data))):
        if filename.endswith(".xls"):
            wb = xlrd.open_workbook(os.path.join(directory_data, filename), logfile=open(os.devnull, 'w'))
            xls = pd.ExcelFile(wb, engine='xlrd')
            for tab_name in tab_names:
                df = pd.read_excel(xls, tab_name, usecols=['IMeasCh2'])
                dfs.append(df)
    t_pd = pd.read_excel(xls, tab_name, usecols=['TimeOutput'])
    return dfs, t_pd

def processWaveformData():
    """
    Reads waveform data and performs basic preprocessing steps.
    """
    I_rd, t, V_rd = rdPRC.rd_data_waveform(root_drc, force_read)
    I_agg_all = I_rd
    V_rd = V_rd[0]
    t = t.flatten()

    I_avg_all = np.mean(I_agg_all, axis=1)
    I_avg_all = np.transpose(I_avg_all)

    print("Shape of the average Current array:", I_avg_all.shape)
    print("Shape of the Time array:", t.shape)

    return I_avg_all, t, V_rd

def calculateEnergy(I_rd, V_rd, t, sample_rate=5):
    """
    Calculates energy consumption based on current and voltage.
    """
    e_all = np.multiply(I_rd, V_rd)
    e_all = np.abs(e_all * sample_rate)

    e_ind = np.reshape(e_all, (I_rd.shape[0], I_rd.shape[2], I_rd.shape[1], I_rd.shape[3]))
    total_e_ind = np.empty((e_ind.shape[0], e_ind.shape[1], e_ind.shape[2]))

    for img_idx in range(e_ind.shape[0]):
        for dvc_idx in range(e_ind.shape[1]):
            for cyc in range(e_ind.shape[2]):
                total_e_ind[img_idx, dvc_idx, cyc] = np.trapz(e_ind[img_idx, dvc_idx, cyc], t)

    avg_energy = np.mean(total_e_ind) * 1e12  # Convert to pJ
    med_energy = np.median(total_e_ind) * 1e12

    print(f"Average energy per device: {avg_energy:.2f} pJ")
    print(f"Median energy per device: {med_energy:.2f} pJ")
    return total_e_ind

def reshapeAndCalculateReadouts(I_avg_all, numofpoints=12, read_start=53, read_interval=88):
    """
    Reshapes current data into readout arrays.
    """
    read_pts = 10
    I_read_avg_out = np.empty((read_pts, numofpoints, I_avg_all.shape[1]))
    for i in range(read_pts):
        start = read_start + read_interval * i
        end = read_start + numofpoints + read_interval * i
        I_read_avg_out[i] = I_avg_all[start:end, :]
    I_read_avg_out = np.transpose(I_read_avg_out, (0, 2, 1))

    print(f"Output array (I_read_avg_out) shape: {I_read_avg_out.shape}")
    return I_read_avg_out

def reshapeToVideo(I_read_avg_out, numofclips=20, aspect_ratio=1.25):
    """
    Reshapes readout arrays to video-like format.
    """
    width = int(np.sqrt(I_read_avg_out.shape[1] / numofclips * aspect_ratio))
    height = int(width / aspect_ratio)

    I_read_avg_out = np.reshape(I_read_avg_out, (I_read_avg_out.shape[0], numofclips, width * height, I_read_avg_out.shape[2]))
    I_read_avg_out = np.transpose(I_read_avg_out, (1, 0, 2, 3))

    print(f"Shape of current readout array: {I_read_avg_out.shape}\nVideo W x H: {width} x {height}")
    return I_read_avg_out, width, height
