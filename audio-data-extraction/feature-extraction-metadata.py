import glob
import librosa
import numpy as np
import pandas as pd


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(filenames):
    rows = len(filenames)
    features, labels = np.zeros((rows,193)), np.zeros((rows,10))
    i = 0
    for fn in filenames:
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            y_col = int(fn.split('/')[4].split('-')[1])
        except:
            print(fn)
        else:
            # One Hot
            features[i] = ext_features
            labels[i, y_col] = 1
            i += 1
    return features, labels

# Rutas
path_main = '../UrbanSound8K'
path_audio = path_main + '/audio'
path_metadata = path_main + '/metadata'

# Metadata
metadata = pd.read_csv(path_metadata + '/UrbanSound8K.csv')

# Construyendo ruta de audio
metadata['src_audio'] = path_audio + '/fold' + metadata['fold'].map(str) + '/' + metadata['slice_file_name']

audio_files = metadata['src_audio'].head(100)
print(len(audio_files))

X, y = parse_audio_files(audio_files)
np.savez('urban-sound-from-metadata', X=X, y=y)
