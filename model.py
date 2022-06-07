import matplotlib.pyplot as plt
import os
from astropy.io import fits
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np

def create_model(channel = 64):

    input_0 = tf.keras.layers.Input(shape=(256, 256, 1))
    reshape_1 = tf.keras.layers.Reshape((256, 256, 1))(input_0)

    patch = tf.keras.layers.Conv2D(filters=1, kernel_size=16, strides=(16, 16))(reshape_1) #, kernel_regularizer='l2'

    con2d_start = tf.keras.layers.Conv2D(filters=channel, kernel_size=(13, 13), activation='relu', padding='same', kernel_regularizer='l2')(reshape_1) #256, 256

    # down 1
    con2d_d11 = tf.keras.layers.Conv2D(filters=channel, kernel_size=(3, 3), activation='relu', padding='same')(con2d_start)
    con2d_d12 = tf.keras.layers.Conv2D(filters=channel * 2, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer='l2')(con2d_d11)
    down_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(con2d_d12)  #128, 128
    drop_d1 = tf.keras.layers.Dropout(0.2)(down_1)

    # down 2
    con2d_d21 = tf.keras.layers.Conv2D(filters=channel * 2, kernel_size=(3, 3), activation='relu', padding='same')(drop_d1)
    con2d_d22 = tf.keras.layers.Conv2D(filters=channel * 4, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer='l2')(con2d_d21)
    down_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(con2d_d22)  #64, 64
    drop_d2 = tf.keras.layers.Dropout(0.2)(down_2)

    # down 3
    con2d_d31 = tf.keras.layers.Conv2D(filters=channel * 4, kernel_size=(3, 3), activation='relu', padding='same')(drop_d2)
    con2d_d32 = tf.keras.layers.Conv2D(filters=channel * 8, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer='l2')(con2d_d31)
    down_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(con2d_d32)  #32, 32
    drop_d3 = tf.keras.layers.Dropout(0.2)(down_3)

    # down 4
    con2d_d41 = tf.keras.layers.Conv2D(filters=channel * 8, kernel_size=(3, 3), activation='relu', padding='same')(drop_d3)
    con2d_d42 = tf.keras.layers.Conv2D(filters=channel * 16, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer='l2')(con2d_d41)
    down_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(con2d_d42)  #16, 16
    drop_d4 = tf.keras.layers.Dropout(0.2)(down_4)

    # middle 1
    con2d_m11 = tf.keras.layers.Conv2D(filters=channel * 16, kernel_size=(3, 3), padding='same')(drop_d4)
    bn_m11 = tf.keras.layers.BatchNormalization()(con2d_m11)
    act_m11 = tf.keras.layers.Activation('relu')(bn_m11)

    cat_m1 = tf.keras.layers.Concatenate(axis=-1)([act_m11, patch])

    con2d_m12 = tf.keras.layers.Conv2D(filters=channel * 16, kernel_size=(3, 3), padding='same')(cat_m1)
    add_m12 = tf.keras.layers.Add()([drop_d4, con2d_m12])
    bn_m12 = tf.keras.layers.BatchNormalization()(add_m12)
    act_m12 = tf.keras.layers.Activation('relu')(bn_m12)

    # middle 2
    con2d_m21 = tf.keras.layers.Conv2D(filters=channel * 16, kernel_size=(3, 3), padding='same')(act_m12)
    bn_m21 = tf.keras.layers.BatchNormalization()(con2d_m21)
    act_m21 = tf.keras.layers.Activation('relu')(bn_m21)

    cat_m2 = tf.keras.layers.Concatenate(axis=-1)([act_m21, patch])

    con2d_m22 = tf.keras.layers.Conv2D(filters=channel * 16, kernel_size=(3, 3), padding='same')(cat_m2)
    add_m22 = tf.keras.layers.Add()([act_m12, con2d_m22])
    bn_m22 = tf.keras.layers.BatchNormalization()(add_m22)
    act_m22 = tf.keras.layers.Activation('relu')(bn_m22)

    # middle 3
    con2d_m31 = tf.keras.layers.Conv2D(filters=channel * 16, kernel_size=(3, 3), padding='same')(act_m22)
    bn_m31 = tf.keras.layers.BatchNormalization()(con2d_m31)
    act_m31 = tf.keras.layers.Activation('relu')(bn_m31)

    cat_m3 = tf.keras.layers.Concatenate(axis=-1)([act_m31, patch])

    con2d_m32 = tf.keras.layers.Conv2D(filters=channel * 16, kernel_size=(3, 3), padding='same')(cat_m3)
    add_m32 = tf.keras.layers.Add()([act_m22, con2d_m32])
    bn_m32 = tf.keras.layers.BatchNormalization()(add_m32)
    act_m32 = tf.keras.layers.Activation('relu')(bn_m32)

    # Fac_0 attentation
    GAP_0 = tf.keras.layers.GlobalAveragePooling2D()(tf.signal.dct(act_m32))
    reshape_Fac_0 = tf.keras.layers.Reshape((-1, ))(GAP_0)
    con1d_0_head0 = tf.keras.layers.Dense(channel * 32, activation='relu')(reshape_Fac_0)
    act_Fac_0_head0 = tf.keras.layers.Dense(channel * 16, activation='sigmoid')(con1d_0_head0)
    reshape_Fac_0_head0 = tf.keras.layers.Reshape((1, 1, -1))(act_Fac_0_head0)
    mul_Fac_0_head0 = tf.keras.layers.Multiply()([act_m32, reshape_Fac_0_head0])

    con1d_0_head1 = tf.keras.layers.Dense(channel * 32, activation='relu')(reshape_Fac_0)
    act_Fac_0_head1 = tf.keras.layers.Dense(channel * 16, activation='sigmoid')(con1d_0_head1)
    reshape_Fac_0_head1 = tf.keras.layers.Reshape((1, 1, -1))(act_Fac_0_head1)
    mul_Fac_0_head1 = tf.keras.layers.Multiply()([act_m32, reshape_Fac_0_head1])

    concat_Fac_0 = tf.keras.layers.Concatenate(axis=-1)([mul_Fac_0_head0, mul_Fac_0_head1])

    # up 1
    up_1 = tf.keras.layers.Conv2DTranspose(filters=channel * 8, kernel_size=(2, 2), strides=(2, 2), activation=tfa.activations.mish)(concat_Fac_0)
    drop_u1 = tf.keras.layers.Dropout(0.2)(up_1)
    con2d_u10 = tf.keras.layers.Conv2D(filters=channel * 8, kernel_size=(3, 3), activation=tfa.activations.mish, padding='same')(drop_u1)

    # Fac_1 attentation
    GAP_1 = tf.keras.layers.GlobalAveragePooling2D()(tf.signal.dct(con2d_u10))
    reshape_Fac_1 = tf.keras.layers.Reshape((-1, ))(GAP_1)
    con1d_1_head0 = tf.keras.layers.Dense(channel * 16, activation='relu')(reshape_Fac_1)
    act_Fac_1_head0 = tf.keras.layers.Dense(channel * 8, activation='sigmoid')(con1d_1_head0)
    reshape_Fac_1_head0 = tf.keras.layers.Reshape((1, 1, -1))(act_Fac_1_head0)
    mul_Fac_1_head0 = tf.keras.layers.Multiply()([con2d_u10, reshape_Fac_1_head0])

    con1d_1_head1 = tf.keras.layers.Dense(channel * 16, activation='relu')(reshape_Fac_1)
    act_Fac_1_head1 = tf.keras.layers.Dense(channel * 8, activation='sigmoid')(con1d_1_head1)
    reshape_Fac_1_head1 = tf.keras.layers.Reshape((1, 1, -1))(act_Fac_1_head1)
    mul_Fac_1_head1 = tf.keras.layers.Multiply()([con2d_u10, reshape_Fac_1_head1])

    concat_Fac_1 = tf.keras.layers.Concatenate(axis=-1)([mul_Fac_1_head0, mul_Fac_1_head1])

    # up 2
    up_2 = tf.keras.layers.Conv2DTranspose(filters=channel * 4, kernel_size=(2, 2), strides=(2, 2), activation=tfa.activations.mish)(concat_Fac_1)
    drop_u2 = tf.keras.layers.Dropout(0.2)(up_2)
    con2d_u20 = tf.keras.layers.Conv2D(filters=channel * 4, kernel_size=(3, 3), activation=tfa.activations.mish, padding='same')(drop_u2)

    # Fac_2 attentation
    GAP_2 = tf.keras.layers.GlobalAveragePooling2D()(tf.signal.dct(con2d_u20))
    reshape_Fac_2 = tf.keras.layers.Reshape((-1, ))(GAP_2)
    con1d_2_head0 = tf.keras.layers.Dense(channel * 8, activation='relu')(reshape_Fac_2)
    act_Fac_2_head0 = tf.keras.layers.Dense(channel * 4, activation='sigmoid')(con1d_2_head0)
    reshape_Fac_2_head0 = tf.keras.layers.Reshape((1, 1, -1))(act_Fac_2_head0)
    mul_Fac_2_head0 = tf.keras.layers.Multiply()([con2d_u20, reshape_Fac_2_head0])

    con1d_2_head1 = tf.keras.layers.Dense(channel * 8, activation='relu')(reshape_Fac_2)
    act_Fac_2_head1 = tf.keras.layers.Dense(channel * 4, activation='sigmoid')(con1d_2_head1)
    reshape_Fac_2_head1 = tf.keras.layers.Reshape((1, 1, -1))(act_Fac_2_head1)
    mul_Fac_2_head1 = tf.keras.layers.Multiply()([con2d_u20, reshape_Fac_2_head1])

    concat_Fac_2 = tf.keras.layers.Concatenate(axis=-1)([mul_Fac_2_head0, mul_Fac_2_head1])

    # up 3
    up_3 = tf.keras.layers.Conv2DTranspose(filters=channel * 2, kernel_size=(2, 2), strides=(2, 2), activation=tfa.activations.mish)(concat_Fac_2)
    drop_u3 = tf.keras.layers.Dropout(0.2)(up_3)
    con2d_u30 = tf.keras.layers.Conv2D(filters=channel * 2, kernel_size=(3, 3), activation=tfa.activations.mish, padding='same')(drop_u3)

    # Fac_3 attentation
    GAP_3 = tf.keras.layers.GlobalAveragePooling2D()(tf.signal.dct(con2d_u30))
    reshape_Fac_3 = tf.keras.layers.Reshape((-1, ))(GAP_3)
    con1d_3_head0 = tf.keras.layers.Dense(channel * 4, activation='relu')(reshape_Fac_3)
    act_Fac_3_head0 = tf.keras.layers.Dense(channel * 2, activation='sigmoid')(con1d_3_head0)
    reshape_Fac_3_head0 = tf.keras.layers.Reshape((1, 1, -1))(act_Fac_3_head0)
    mul_Fac_3_head0 = tf.keras.layers.Multiply()([con2d_u30, reshape_Fac_3_head0])

    con1d_3_head1 = tf.keras.layers.Dense(channel * 4, activation='relu')(reshape_Fac_3)
    act_Fac_3_head1 = tf.keras.layers.Dense(channel * 2, activation='sigmoid')(con1d_3_head1)
    reshape_Fac_3_head1 = tf.keras.layers.Reshape((1, 1, -1))(act_Fac_3_head1)
    mul_Fac_3_head1 = tf.keras.layers.Multiply()([con2d_u30, reshape_Fac_3_head1])

    concat_Fac_3 = tf.keras.layers.Concatenate(axis=-1)([mul_Fac_3_head0, mul_Fac_3_head1])

    # up 4
    up_4 = tf.keras.layers.Conv2DTranspose(filters=channel, kernel_size=(2, 2), strides=(2, 2), activation=tfa.activations.mish)(concat_Fac_3)
    drop_u4 = tf.keras.layers.Dropout(0.2)(up_4)
    con2d_u40 = tf.keras.layers.Conv2D(filters=channel, kernel_size=(3, 3), activation=tfa.activations.mish, padding='same')(drop_u4)

    # Fac_4 attentation
    GAP_4 = tf.keras.layers.GlobalAveragePooling2D()(tf.signal.dct(con2d_u40))
    reshape_Fac_4 = tf.keras.layers.Reshape((-1, ))(GAP_4)
    con1d_4_head0 = tf.keras.layers.Dense(channel * 2, activation='relu')(reshape_Fac_4)
    act_Fac_4_head0 = tf.keras.layers.Dense(channel, activation='sigmoid')(con1d_4_head0)
    reshape_Fac_4_head0 = tf.keras.layers.Reshape((1, 1, -1))(act_Fac_4_head0)
    mul_Fac_4_head0 = tf.keras.layers.Multiply()([con2d_u40, reshape_Fac_4_head0])

    con1d_4_head1 = tf.keras.layers.Dense(channel * 2, activation='relu')(reshape_Fac_4)
    act_Fac_4_head1 = tf.keras.layers.Dense(channel, activation='sigmoid')(con1d_4_head1)
    reshape_Fac_4_head1 = tf.keras.layers.Reshape((1, 1, -1))(act_Fac_4_head1)
    mul_Fac_4_head1 = tf.keras.layers.Multiply()([con2d_u40, reshape_Fac_4_head1])

    concat_Fac_4 = tf.keras.layers.Concatenate(axis=-1)([mul_Fac_4_head0, mul_Fac_4_head1])

    cat_out = tf.keras.layers.Concatenate(axis=-1)([concat_Fac_4, reshape_1])

    con2d_output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')(cat_out)

    output_0 = tf.keras.layers.Reshape((256, 256, 1))(con2d_output)

    model = tf.keras.Model(input_0, output_0)
    return model