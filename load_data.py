import matplotlib.pyplot as plt
import os
from astropy.io import fits
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np


top = 1500.
tail = -1500.

def normal_(data_list):
    return data_list / top

def dis_normal_(data_list):
    return data_list * top

def get_dataset(input_path = "data/image/", label_path = "data/label/", output_path = "data/output/"):
    input_data_row = []
    for name in os.listdir(input_path):
        fits_file = fits.open(input_path + name)
        fits_data = fits_file[0].data
        input_data_row.append((name, fits_data))

    input_data_sort = sorted(input_data_row,key = lambda x:x[0])

    input_data = []
    for i in input_data_sort:
        input_data.append(i[1])

    input_data = tf.Variable(initial_value=input_data, dtype=tf.float32)

    label_data_row = []
    for name in os.listdir(label_path):
        fits_file = fits.open(label_path + name)
        fits_data = fits_file[0].data
        label_data_row.append((name, fits_data))

    label_data_sort = sorted(label_data_row,key = lambda x:x[0])

    label_data = []
    for i in label_data_sort:
        label_data.append(i[1])

    label_data = tf.Variable(initial_value=label_data, dtype=tf.float32)

    input_data = tf.clip_by_value(input_data, clip_value_min=-1500., clip_value_max=1500.)
    label_data = tf.clip_by_value(label_data, clip_value_min=-1500., clip_value_max=1500.)

    input_data = normal_(input_data)
    label_data = normal_(label_data)

    input_data = tf.reshape(input_data, shape=(-1, 256, 256, 1))
    label_data = tf.reshape(label_data, shape=(-1, 256, 256, 1))

    x_train, x_test = input_data[0:-76], input_data[-76:]
    y_train, y_test = label_data[0:-76], label_data[-76:]

    # x_train, y_train = tf.concat([x_train, tfa.image.rotate(x_train, 90)], 0), tf.concat([y_train, tfa.image.rotate(y_train, 90)], 0)
    # x_train, y_train = tf.concat([x_train, tfa.image.rotate(x_train, 180)], 0), tf.concat([y_train, tfa.image.rotate(y_train, 180)], 0)

    return x_train, y_train, x_test, y_test