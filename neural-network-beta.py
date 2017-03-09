import glob
import os
import librosa
import librosa.display

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
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
number_of_samples = 10


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
        plt.subplot(number_of_samples, 1, i)
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
        plt.subplot(number_of_samples, 1, i)
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
        plt.subplot(number_of_samples, 1, i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1

    fig.canvas.set_window_title('Figura 3: Tono')
    plt.show()


# =================================================================================================
# Extracción de características de los sonidos para nuestra red neuronal
# =================================================================================================
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(scr_audio):
    features = np.empty((0, 193))
    for index in range(len(scr_audio)):
        print(scr_audio.iloc[index])
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(scr_audio.iloc[index])
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features,ext_features])

    return np.array(features)




# Rutas
path_main = 'UrbanSound8K'
path_audio = path_main + '/audio'
path_metadata = path_main + '/metadata'

# Metadata
metadata = pd.read_csv(path_metadata + '/UrbanSound8K.csv')
metadata_test = pd.read_csv(path_metadata + '/UrbanSound8K-Test.csv')

# Construyendo ruta de audio
metadata['src_audio'] = path_audio + '/fold' + metadata['fold'].map(str) + '/' + metadata['slice_file_name']
metadata_test['src_audio'] = path_audio + '/fold' + metadata_test['fold'].map(str) + '/' + metadata_test['slice_file_name']

# Load data - Etiquetas
# One Hot Encoder
# https://www.tensorflow.org/versions/master/api_docs/python/array_ops/slicing_and_joining#one_hot
# Classification using Multilayer Neural Network
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
session = tf.Session()

# Variables para seleccion de muestras
# samples_total = len(metadata)
samples_total = number_of_samples
categories_total = 3
# print('Total de muestras')
# print(samples_total)

# TRAIN
labels = metadata['classID'].head(samples_total)
labels = labels.values.reshape(1, samples_total)

#print(labels)
labels = tf.one_hot(labels, depth=categories_total, dtype=tf.float32)
labels = tf.cast(labels, tf.float32)  # WTF
print('Label Train')
print(session.run(labels))


features = parse_audio_files(metadata['src_audio'].head(samples_total))

# SAMPLES FOR TEST
labels_test = metadata_test['classID']
labels_test = labels_test.values.reshape(1, 2)
labels_test = tf.one_hot(labels_test, depth=categories_total, dtype=tf.float32)
labels_test = tf.cast(labels_test, tf.float32)  # WTF
print('Label Test')
print(session.run(labels_test))
features_test = parse_audio_files(metadata_test['src_audio'])

# set values
train_x = features
train_y = labels

test_x = features_test
test_y = labels_test

# Classification
training_epochs = 1000
n_dim = features.shape[1]
print(n_dim)
n_classes = 10
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

# Variables del modelo
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean=0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)


# Costo
cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None

session.run(tf.global_variables_initializer())
# print('train_x')
# print(train_x)
# print('train_y')
# print(type(train_y.eval()))
for epoch in range(training_epochs):
    cost = session.run(optimizer, {X: train_x, Y: int(train_y.eval(session=session).reshape(10,None))})
    cost_history = np.append(cost_history, cost)

sys.exit()
y_pred = session.run(tf.argmax(y_, 1), {X: test_x})
y_true = session.run(tf.argmax(test_y, 1))


fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.axis([0, training_epochs, 0, np.max(cost_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
print("F-Score:", round(f, 3))
