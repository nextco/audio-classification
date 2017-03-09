import librosa
import numpy as np
np.set_printoptions(threshold=np.nan)

import tensorflow as tf
import sys

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Own Timing WTF import
import importlib
module_with_hyphen = importlib.import_module('audio-data-extraction.utils.timing')
module_with_hyphen.main()

# Data
sound_data = np.load('audio-data-extraction/urban-sound-from-metadata.npz')
X_data = sound_data['X']
y_data = sound_data['y']

# Explorando los datos
print(X_data.shape, y_data.shape)
print(y_data)

# Definir porcentaje de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_sub, X_test, y_sub, y_test = train_test_split(X_data, y_data, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2)
print('Porcentaje de entrenamiento y prueba')
print(len(X_train), len(X_val), len(X_test), len(y_train), len(y_val), len(y_test))

# Variables del modelo
training_epochs = 6000
n_dim = 193  # El modelo tiene 193 caracteristicas extraidas con librosa
n_classes = 10
n_hidden_units_one = 300
n_hidden_units_two = 200
n_hidden_units_three = 100
learning_rate = 0.1
sd = 1 / np.sqrt(n_dim)

# Variables de Tf
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name="w1")
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name="b1")
h_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name="w2")
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name="b2")
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)

W_3 = tf.Variable(tf.random_normal([n_hidden_units_two, n_hidden_units_three], mean=0, stddev=sd), name="w3")
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean=0, stddev=sd), name="b3")
h_3 = tf.nn.sigmoid(tf.matmul(h_2, W_3) + b_3)

W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="w")
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name="b")

# El modelo propuesto
y_ = tf.nn.softmax(tf.matmul(h_3, W) + b)


# Entrenar
cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# Optimizar
cost_history = np.empty(shape=[1], dtype=float)

session = tf.Session()
session.run(tf.global_variables_initializer())
for epoch in range(training_epochs):
    _, cost = session.run([optimizer, cost_function], feed_dict={X: X_sub, Y: y_sub})
    cost_history = np.append(cost_history, cost)

print('Validation accuracy: ',round(session.run(accuracy, feed_dict={X: X_test, Y: y_test}), 3))
saver.save(session, "./model-ckpt/trained_py_6k.ckpt")

# Imprimir costo de Entrenamiento
plt.plot(cost_history)
plt.show()
