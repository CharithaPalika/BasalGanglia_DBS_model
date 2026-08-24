from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.signal import hilbert
import os
from scipy.signal import welch
from scipy.stats import entropy
import pickle as pkl

def spectral_entropy(signal, fs=1.0, nperseg=100000, fmax=50, normalize=True):
    # Compute the power spectral density (PSD)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    
    # Mask the PSD to limit the frequency range to 0-100 Hz
    mask = (freqs >= 0) & (freqs <= fmax)
    freqs = freqs[mask]
    psd = psd[mask]
    
    # Normalize the PSD to form a probability distribution
    psd /= np.sum(psd)
    psd = np.where(psd == 0, 1e-12, psd)  # Avoid log(0)
    
    # Compute the spectral entropy (Shannon entropy)
    se = entropy(psd)
    
    if normalize:
        se /= np.log(len(psd))  # Normalize to [0, 1]
    return se


def APC(signal):
    signal = np.array(signal)   
    corr = np.corrcoef((signal))
    avg_corr = np.mean(corr[np.triu_indices(4, k = 1)])
    return avg_corr


def phase_sync(signal):
    analytic_signals = hilbert(signal, axis=1)
    phases = np.angle(analytic_signals)
    N, T = phases.shape
    plv_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            phase_diff = phases[i] - phases[j]
            plv = np.abs(np.sum(np.exp(1j * phase_diff)) / T)
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv  
    avg_phase_sync = np.mean(plv_matrix[np.triu_indices(4, k = 1)])
    return avg_phase_sync