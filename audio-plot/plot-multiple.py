import glob
import os
import librosa
import librosa.display

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
plt.style.use('ggplot')

# Fuentes por defecto para windows
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.serif'] = 'sans-serif'
plt.rcParams['font.monospace'] = 'Courier New'

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13

import sys

# Numero de muestras a imprimir
number_of_plots = 5


def load_sound_files(file_paths):
    #print('load_sound_files')
    raw_sounds = []
    for fp in file_paths:
        x, sr = librosa.load(fp)
        # print(x)
        raw_sounds.append(x)
    return raw_sounds


def plot_waves(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(16, 11))

    # Formatear espacios entre subfiguras
    plt.subplots_adjust(hspace=1.0)

    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(number_of_plots, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    fig.canvas.set_window_title('Figura 1: Ondas')
    plt.show()


def plot_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(16, 11))
    plt.subplots_adjust(hspace=1.0)

    for n, f in zip(sound_names,raw_sounds):
        plt.subplot(number_of_plots, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    fig.canvas.set_window_title('Figura 2: Frecuencia')
    plt.show()


def plot_log_power_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(16, 11))
    plt.subplots_adjust(hspace=1.0)

    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(number_of_plots, 1, i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1

    fig.canvas.set_window_title('Figura 3: Tono')
    plt.show()

# Rutas
path_main = '../UrbanSound8K'
path_audio = path_main + '/audio'
path_metadata = path_main + '/metadata'

# Metadata
metadata = pd.read_csv(path_metadata + '/UrbanSound8K.csv')

# Construyendo ruta de audio
metadata['src_audio'] = path_audio + '/fold' + metadata['fold'].map(str) + '/' + metadata['slice_file_name']

# For print sample plots
sound_names = metadata['class'].head(number_of_plots)
raw_sounds = load_sound_files(metadata['src_audio'].head(number_of_plots))
plot_waves(sound_names, raw_sounds)
plot_specgram(sound_names, raw_sounds)
plot_log_power_specgram(sound_names, raw_sounds)
