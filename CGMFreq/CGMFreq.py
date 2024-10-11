# +
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
from datetime import datetime, timedelta
import pickle
import numpy as np

from scipy.signal import butter, lfilter, freqz, filtfilt,welch,find_peaks,periodogram, spectrogram
import scipy.stats as stats
from scipy.fft import fft,fftfreq

import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings('ignore')


# -

# Preprocess
def process_cgm(df,CGM_column,time_column, fs):  
    df[time_column]=pd.to_datetime(df[time_column])
    signal = df[CGM_column]
    time = df[time_column]

    time_diff = (df['Time'].iloc[-1] - df['Time'].iloc[0])
    missing_readings= round(time_diff.seconds/(1/fs)-len(df),1)
    print(f'Warning: this data is missing approximately {missing_readings} CGM readings ')
    return signal.values, time


def filter_cgm(y,t,cutoff=.0005,fs = 1/300,show_fig = 1,fig_name = 'filtered_signal'):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    def filter_lowpass(data, cutoff_freq, fs, order=6):
        b, a = butter(order, cutoff_freq, fs=fs, btype='low') 
        y = filtfilt(b, a, data)
        return y
    
    data = np.asarray(y)-np.mean(y) 
    y_filt = filter_lowpass(data, cutoff, fs)
    if show_fig==1:
        plt.figure(figsize=(5,3))
        plt.plot_date(t, data, color=colors[2] , label='data')
        plt.plot_date(t, y_filt, 'g-', linewidth=2, color = 'black',label='filtered data')
        plt.xlabel('Time of Day [HH:MM]')
        plt.grid()
        plt.legend()
        plt.subplots_adjust(hspace=0.35)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.title('Time Series Signal', fontsize=14, fontweight='bold',color=colors[2])
        plt.savefig(fig_name,dpi=400)
        plt.title('Time Series Signal', fontsize=14, fontweight='bold',color=colors[2])
        plt.show()
    return y_filt + np.mean(y)


def visualize( signal, time, fs = 1/300 ,nperseg=32,show_figs=True,fig_name= 'Visualize',window = 'hann'):   ## FFT Features
    signal = signal-np.mean(signal) 
    N = len(signal)
    frequencies_psd, power_spectrum = welch(signal, fs=fs, nperseg=nperseg, window=window)
    fft_result = fft(signal)
    frequencies = np.fft.fftfreq(len(signal), d= 1/fs)
    
    if show_figs == True:
        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        axs[0].plot(frequencies[:N//2], np.abs(fft_result[:N//2]), color=colors[0])
        axs[0].set_xlim([0, .0003])
        axs[0].set_xlabel('Frequency [Hz]')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Fast Fourier Transform', fontweight='bold', color=colors[0])

        axs[1].plot(frequencies_psd, power_spectrum, color=colors[1])
        axs[1].set_xlabel('Frequency [Hz]')
        axs[1].set_ylabel('Power')
        axs[1].set_xlim([0, .0003])
        axs[1].set_title('Power Spectral Density', fontweight='bold', color=colors[1])

        axs[2].plot_date(time, signal, color=colors[2])
        axs[2].set_title('CGM Data', fontweight='bold', color=colors[2])
        axs[2].set_xlabel('Time of Day [HH:MM]')
        axs[2].set_ylabel('Reading [mg/dL]')
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        plt.tight_layout()
        plt.savefig(fig_name, dpi=400)
        plt.show()


# +
def freq_feats(signal,fs = 1/300,nperseg=32,window = 'hann'):   ## FFT Features
    ################################################
 
    # function takes in signal with daily pre-processed CGM data as input, outputs a dict with features

    ################################################
    
    signal = signal-np.mean(signal) 
    N = len(signal)
    frequencies_psd, power_spectrum = welch(signal, fs, nperseg=nperseg, window=window)
    fft_result = fft(signal)
    frequencies = np.fft.fftfreq(len(signal), d= 1/fs)

    feat_df = {}
    feat_df['fft_max_amplitude']  = np.max(np.abs(fft_result))
    feat_df['fft_dominant_frequency' ]= frequencies[np.argmax(np.abs(fft_result))]

    feat_df['psd_max_amplitude' ] = np.max(power_spectrum)
    feat_df['psd_dominant_frequency'] = frequencies_psd[np.argmax(power_spectrum)]

    cumulative_power = np.cumsum(power_spectrum)
    energy_threshold = np.sum(power_spectrum)*.75
    index = np.where(cumulative_power >= energy_threshold)[0][0]
    frequency_at_threshold = frequencies_psd[index]
    feat_df['psd75_frequency'] = frequency_at_threshold


    cumulative_fft = np.cumsum(np.abs(fft_result))
    energy_threshold = np.sum(np.abs(fft_result)[:N//2])*.75
    index = np.where(cumulative_fft >= energy_threshold)[0][0]
    frequency_at_threshold = frequencies[index]
    feat_df['fft75_frequency'] = frequency_at_threshold
    
    # Get PSD db
    psd_dB = 10*np.log10(power_spectrum)
    max_psd_dB = np.max(psd_dB)
    threshold_dB = max_psd_dB-3
    
    indices_within_threshold = np.where(psd_dB>=threshold_dB)[0]
    f_lower = frequencies_psd[indices_within_threshold[0]]
    f_higher = frequencies_psd[indices_within_threshold[-1]]
    bandwidth = f_higher-f_lower
    feat_df['bandwidth']=bandwidth


    # Extract heights of the peaks
    peaks, _= find_peaks(np.abs(fft_result))
    peak_heights = np.abs(fft_result)[peaks]
    sorted_peak_indices = np.argsort(peak_heights)[-2]  
    second_highest_peak = frequencies[peaks[sorted_peak_indices]]
    second_peak_height = peak_heights[sorted_peak_indices]

    feat_df['fft_second_peak_mag'] =second_peak_height
    feat_df['fft_second_peak_freq']=second_highest_peak

    return feat_df

def time_feats( signal):   ## FFT Features
    time_df = {}
    time_df['Average']  = np.mean(signal)
    time_df['Std']  = np.std(signal)
    time_df['CV' ]= np.std(signal)/np.mean(signal)
    time_df['Minimum'] = np.min(signal)
    time_df['Maximum'] = np.max(signal)
    tbh= signal[signal<180]
    print(tbh)
    time_df['TIR'] = len(tbh[tbh>70])/(len(signal))
    time_df['TAR 1 (>180)'] = len(signal[signal>180])/(len(signal))
    time_df['TAR 2 (>250)'] = len(signal[signal>250])/(len(signal))
    time_df['TBR 1 (<70)'] = len(signal[signal<70])/(len(signal))
    time_df['TBR 2 (<54)'] = len(signal[signal<54])/(len(signal))
    return time_df


# +
def CGMspectrogram(signal,time, fs,nperseg, fig_name = 'spectrogram'):
    N = len(signal)
    t = time
    t_mpl = mdates.date2num(t)

    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=4.5)  
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    ax1.plot_date(t_mpl, signal, 'o', label='Signal', markersize=3,color=colors[2])
    ax1.set_ylabel('Reading [mg/dL]')
    ax1.set_title('Time Series Signal', fontsize=14, fontweight='bold',color=colors[2])

    frequencies, times, Sxx = spectrogram(signal, fs=fs,nperseg=nperseg,)  
    times_mpl = mdates.date2num([t[0] + timedelta(seconds=i) for i in times]) 
    ax1.set_xlabel('Time of Day [HH:MM]')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    cax = ax2.pcolormesh(times_mpl, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time of Day [HH:MM]')
    ax2.set_title('Spectrogram', fontsize=14, fontweight='bold',color=colors[0])
    ax2.set_ylim([0,.0001])
    
    plt.colorbar(cax, label='Intensity [dB]',ax=ax2,orientation="horizontal")
    ax2.set_xlabel('Time of Day [HH:MM]')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()
    plt.subplots_adjust(hspace=0.6) 
    ax1.set_xlabel('Time of Day [HH:MM]')

    plt.savefig(fig_name, dpi=400)
    plt.show()
    return frequencies, times, Sxx 


