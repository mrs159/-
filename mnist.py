# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 18:33:36 2020

@author: meng
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import layers, datasets, optimizers, Sequential, metrics
import datetime
def preprocess(x,y):
  x = tf.cast(x, dtype=tf.float32)/255.
  x = tf.reshape(x, [-1])
  y = tf.cast(y, dtype=tf.int32)
  y = tf.one_hot(y, depth=10)
  return x,y

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('log', current_time)
summary_writer = tf.summary.create_file_writer(log_dir)

#load data
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
#print(x_train.shape,y_train.shape)
#print(x_test.shape,y_test.shape)
db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db = db.map(preprocess).shuffle(60000).batch(200)
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(200)
#build network
network = Sequential([
layers.Dense(512, activation=tf.nn.relu),
layers.Dense(256, activation=tf.nn.relu),
layers.Dense(128, activation=tf.nn.relu),
layers.Dense(64, activation=tf.nn.relu),
layers.Dense(32, activation=tf.nn.relu),
layers.Dense(10)
])
network.build(input_shape=[None, 28*28])
network.summary()
#input para

network.compile(optimizer=optimizers.Adam(lr=1e-2),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                          write_graph=True, write_images=False)


#run network
history = network.fit(db, epochs=20, validation_data=db_test,
                      validation_freq=1, callbacks=[tensorboard])

network.evaluate(db_test)
#network.save('model', save_format=tf)