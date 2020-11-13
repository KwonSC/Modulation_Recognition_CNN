import os,random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy
from tensorflow.keras.utils import HDF5Matrix
from sklearn.model_selection import train_test_split
filename = "GOLD_XYZ_OSC.0001_1024.hdf5"
mods = np.array(['32PSK', '16APSK', '32QAM', 'FM', 'GMSK','32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM'])
data = h5py.File('new_data_01.h5', 'r') 
x = data.get('x').value
y = data.get('y').value
z = data.get('z').value
tr_x, ts_x, tr_y, ts_y = train_test_split(x, y, test_size=0.3, random_state=444)
tr_z, ts_z = train_test_split(z, test_size=0.3, random_state=444)
np.random.seed(2020)
in_shp = list(tr_x.shape[1:])
num_classes = len(mods)
classes = mods
dr = 0.5 # dropout rate (%)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Reshape(in_shp + [1], input_shape=in_shp))
model.add(tf.keras.layers.Conv2D(32, (2, 2), padding='valid', activation="relu", input_shape=[1024, 2, 1]))
#model.add(tf.keras.layers.Dropout(dr))
model.add(tf.keras.layers.Reshape([1023, 32]))
#model.add(tf.keras.layers.Dropout(dr))
model.add(tf.keras.layers.Conv1D(64, 3, strides=2, padding="valid", activation="relu"))
#model.add(tf.keras.layers.Dropout(dr))
model.add(tf.keras.layers.Conv1D(128, 3, strides=2, padding="valid", activation="relu"))
#model.add(tf.keras.layers.Dropout(dr))
model.add(tf.keras.layers.Conv1D(256, 3, strides=2, padding="valid", activation="relu"))
#model.add(tf.keras.layers.Dropout(dr))
model.add(tf.keras.layers.Conv1D(128, 3, strides=2, padding="valid", activation="relu"))
#model.add(tf.keras.layers.Dropout(dr))
model.add(tf.keras.layers.Conv1D(64, 3, strides=2, padding="valid", activation="relu"))
#model.add(tf.keras.layers.Dropout(dr))
model.add(tf.keras.layers.Conv1D(32, 3, strides=2, padding="valid", activation="relu"))
#model.add(tf.keras.layers.Dropout(dr))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(dr))
model.add(tf.keras.layers.Dense(num_classes))
model.add(tf.keras.layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
nb_epoch = 100 #number of epochs to train on
batch_size = 1024
history1 = model.fit(tr_x,
    tr_y,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=(ts_x, ts_y))
# we re-load the best weights once training is finished
score1 = model.evaluate(ts_x, ts_y, verbose=1, batch_size=batch_size)
print(score1)
model.save('new_11_13_01.h5')
data = h5py.File('new_data_02.h5', 'r') 
x = data.get('x').value
y = data.get('y').value
z = data.get('z').value
tr_x, ts_x, tr_y, ts_y = train_test_split(x, y, test_size=0.3, random_state=1)
tr_z, ts_z = train_test_split(z, test_size=0.3, random_state=1)
history2 = model.fit(tr_x,
    tr_y,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=(ts_x, ts_y))
# we re-load the best weights once training is finished
score2 = model.evaluate(ts_x, ts_y, verbose=1, batch_size=batch_size)
print(score2)
model.save('new_11_13_02.h5')