import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt

# Set Config
# http://matplotlib.org/users/style_sheets.html
# print(plt.style.available)
plt.style.use('ggplot') # ggplot is a plotting system for R, copy the style

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

# Perro ladrando
src_audio = '../UrbanSound8K/audio/fold1/7383-3-0-0.wav'

# Cargar Info de Audio
raw_data, src = librosa.load(src_audio)

# Plot
# figsize - w,h tuple in inches
fig = plt.figure(figsize=(16, 11))

# Formatear espacios entre subfiguras
# Source http://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
plt.subplots_adjust(hspace=1.0)

plt.subplot(3, 1, 1)
librosa.display.waveplot(np.array(raw_data), sr=22050)
plt.title('Ondas - Clásico -  Idea de donde suena más fuerte - Juntado')

plt.subplot(3, 1, 2)
plt.specgram(np.array(raw_data), Fs=22050)
plt.title('Frecuencia Espectral - En que frecuencias se escucha')

plt.subplot(3, 1, 3)
D = librosa.logamplitude(np.abs(librosa.stft(raw_data))**2, ref_power=np.max)
librosa.display.specshow(D, x_axis='time', y_axis='log')
plt.title('Tono Espectral - Indicador si sube o baja')

# Set Main Window Title
fig.canvas.set_window_title('Graficando Audio con librosa')
plt.show()


# Wave
