"""
Filename:        ChG-PRD_Model.py
Author:          Jia-Yi LI
Last Modified:   2024-12-20
Version:         1.0
Description:     Fitting physical reservoir device model based on device data.
License:         MIT License
                 Copyright (c) 2024 Nanyang Technological University
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import Data.rd_PRC as rdPRC

def load_data(root_drc, drc_I):
    """Load waveform and sweep data."""
    I_rd_wfm, t_rd_wfm, _ = rdPRC.rd_data_waveform(drc_I, force_read=False)
    I_rd_wfm = np.negative(I_rd_wfm)

    I_rd, t_rd = rdPRC.rd_data_sweep(root_drc)
    I_rd = np.negative(I_rd)

    return I_rd_wfm, t_rd_wfm, I_rd, t_rd

def calculate_avg_std(I_rd_wfm, read_start=53, read_interval=88, numofpoints=12, read_pts=14):
    """Calculate mean and standard deviation for data points."""
    num_of_tests = I_rd_wfm.shape[0]
    I_rd_avg_out = np.empty((num_of_tests, read_pts, I_rd_wfm.shape[1], numofpoints))
    
    for j in range(num_of_tests):
        for i in range(read_pts):
            start = read_start + read_interval * i
            end = start + numofpoints
            I_rd_avg_out[j, i] = I_rd_wfm[j, :, start:end]

    I_rd_pts_mean = np.mean(np.mean(I_rd_avg_out, axis=3), axis=2)
    I_rd_pts_std = np.std(np.mean(I_rd_avg_out, axis=3), axis=2)
    
    return I_rd_pts_mean, I_rd_pts_std

def fit_linear_model(I_aft, I_bfr):
    """Fit a linear regression model to the data."""
    x = np.array([I_aft[i].mean() for i in range(I_aft.shape[0])]).reshape(-1, 1)
    y = np.array([I_bfr[i].mean() for i in range(I_aft.shape[0])]).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    r2 = r2_score(y, y_pred)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    return slope, intercept, r2, y_pred

def model_cal(A, y0, tau, t, t0=0, d2d_var=0):
    """Compute exponential decay model output."""
    tau_var = np.clip(tau * (1 + np.random.normal(loc=0, scale=d2d_var, size=1)), a_min=1e-9, a_max=1e-4)
    A_var = np.clip(A * (1 + np.random.normal(loc=0, scale=d2d_var, size=1)), a_min=-1e-4, a_max=1e-4)
    y0_var = np.clip(y0 * (1 + np.random.normal(loc=0, scale=d2d_var, size=1)), a_min=-1e-4, a_max=1e-4)
    k = 1 / tau_var

    return A_var * np.exp(-k * (t - t0)) + y0_var

def current_cal_binary(pulses, tau_params, var=0.05):
    """Calculate current for a sequence of binary pulses."""
    t_seq = np.arange(1, 221, 5, dtype=np.float16) / 1e9
    dummy_seq = np.arange(221, 441, 5) / 1e9
    I0 = 1e-8
    current = np.array([])

    for pulse in pulses:
        if pulse == 0:
            temp_current = model_cal(I0, y0=1e-9, tau=tau_params[2, 0], t=np.concatenate((t_seq, dummy_seq)), d2d_var=var)
        elif pulse == 1:
            temp_current = model_cal(A_params[2, 0], y0=1e-9, tau=tau_params[2, 0], t=t_seq, d2d_var=var)
            temp_current = np.concatenate((np.full(44, np.nan), temp_current))
        else:
            raise ValueError("Pulse value out of range")

        current = np.concatenate((current, temp_current))
        I0 = current[-1]

    return current

def predict_binary_model(bit_length, pulse_idx, tau_params, var=0):
    """Generate predicted currents based on binary model."""
    t_seq = np.arange(1, bit_length * 88 + 1) * 5 / 1e9
    pulse = np.array([int(x) for x in bin(pulse_idx)[2:].zfill(bit_length)])
    I = current_cal_binary(pulse, tau_params, var)

    return I, t_seq

if __name__ == "__main__":
    root_drc = "/path/to/root"
    drc_I = "/path/to/I"

    # Example execution
    I_rd_wfm, t_rd_wfm, I_rd, t_rd = load_data(root_drc, drc_I)
    I_mean, I_std = calculate_avg_std(I_rd_wfm)
    
    # Placeholder for actual I_aft and I_bfr
    I_aft = np.random.random((10, 10))
    I_bfr = np.random.random((10, 10))
    
    slope, intercept, r2, y_pred = fit_linear_model(I_aft, I_bfr)
    print(f"Linear model: y = {slope:.2e}x + {intercept:.2e} (R^2 = {r2:.2f})")

    # Predict currents for binary pulses
    I, t_seq = predict_binary_model(20, 1095, np.random.random((3, 256)), var=0.05)
