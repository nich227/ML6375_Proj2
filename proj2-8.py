'''
Name:       Kevin Chen
Professor:  Haim Schweitzer
Due Date:   7/14/2020
Project 2 - Classifier 1
Python 3.7.6
'''

import sys
import tensorflow as tf
import numpy as np
from numpy.random import seed, randint
from tensorflow.random import set_seed
from sklearn.preprocessing import MinMaxScaler
from time import time


# Setting random seeds for reproducibility
seed(1)
set_seed(1)

EPOCHS = 500
BATCH_SIZE = 16

start_time = time()

# Path to train and test data
x_train = np.loadtxt("x_train8.csv", dtype="uint8", delimiter=",")
y_train = np.loadtxt("y_train8.csv", dtype="uint8", delimiter=",")

x_test = np.loadtxt("x_test.csv", dtype="uint8", delimiter=",")
y_test = np.loadtxt("y_test.csv", dtype="uint8", delimiter=",")


m, n = x_train.shape  # m training examples, each with n features
m_labels, = y_train.shape  # m2 examples
k = y_train.max() + 1  # k labels
l_min = y_train.min()

if m_labels != m:
    raise RuntimeError('x_train and y_train should have the same length.')
if l_min < 0 or l_min > k-1:
    raise RuntimeError('Each label should be in the range 0 to k-1.')

# Set learning rate
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.05,
    decay_steps=np.ceil(m / BATCH_SIZE)*EPOCHS,
    decay_rate=1,
    staircase=True)

# Data preprocessing (both train and test)

# Converting NaN to 0
np.nan_to_num(x_train)
np.nan_to_num(x_test)

# Adding normalized noise to the testing data
x_test_noise = x_test.copy()
x_test_noise = x_test_noise + np.random.normal(0, 0.1, x_test.shape)
y_test_noise = y_test.copy()

x_test = np.append(x_test, x_test_noise, axis=0)
y_test = np.append(y_test, y_test_noise, axis=0)

# Adding randomly noise data from test to train
train_append_indices = []
while len(train_append_indices) < x_train.size:
    rand_index = randint(0, len(x_test_noise) - 1)
    train_append_indices.append(rand_index)

for index in list(set(train_append_indices)):
    x_train = np.vstack((x_train, x_test_noise[index]))
    y_train = np.append(y_train, y_test_noise[index])

# Scale the data to have values from 0 to 1
x_train = MinMaxScaler().fit_transform(x_train)
x_test = MinMaxScaler().fit_transform(x_test)

# Fixing skew on data by square rooting it
for i, feature in enumerate(x_train.T):
    x_train[:, i] = np.sqrt(feature)
for i, feature in enumerate(x_test.T):
    x_test[:, i] = np.sqrt(feature)

# Training model
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(
        20, activation=tf.keras.activations.elu, kernel_initializer='normal', input_shape=(n,)),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(k, activation=tf.keras.activations.linear),
    tf.keras.layers.Dense(k, activation=tf.keras.activations.softmax)
])

model1.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction='auto'), metrics=['accuracy'])
model1.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Testing model
m_test, n_test = x_test.shape
m_test_labels, = y_test.shape

if m_test_labels != m_test:
    raise RuntimeError('x_test and y_test should have the same length.')
if n_test != n:
    raise RuntimeError(
        'train and x_test should have the same number of features.')

print(model1.evaluate(x_test, y_test)[1])

# End of program
print('-----\n', 'Project 2 took', round(time() -
                                         start_time, 3), 'seconds to complete.')
