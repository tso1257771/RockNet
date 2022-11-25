import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.activations import softmax, sigmoid
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Attention, LayerNormalization, add
from tensorflow.keras.layers import AveragePooling1D

# general parameters
activation="relu"
out_activation="softmax"
dropout_rate=0.1
batchnorm=True
kernel_init = 'he_uniform'
kernel_regu = None
RRconv_time = 2
padding = 'same'

# seismogram parameters
ts_input_size=(6000, 3)
ts_nb_filters=[6, 12, 18, 24, 30]
ts_kernel_size=7
ts_strides=5
ts_upsize=5

# spectrogram parameters.
spec_input_size= (51, 601, 2)
spec_nb_filters = [6, 12, 18, 24, 30]
spec_kernel_size = (3, 3)
spec_strides = (3, 3)
spec_upsize = 3


# time series operation blocks
def ts_conv1d(ts_nb_filter, ts_strides=None):
    if ts_strides:
        return Conv1D(
            ts_nb_filter,
            ts_kernel_size,
            padding=padding,
            strides=ts_strides,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_regu,
        )
    else:
        return Conv1D(
            ts_nb_filter,
            ts_kernel_size,
            padding=padding,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_regu,
        )

def ts_conv_unit(inputs, ts_nb_filter, ts_strides, name=None):

    if ts_strides != None:
        u = ts_conv1d(ts_nb_filter, ts_strides=ts_strides)(inputs)
        if batchnorm:
            u = BatchNormalization()(u)
        u = Activation(activation)(u)
        if dropout_rate:
            u = Dropout(dropout_rate, name=name)(u)

    else:
        u = ts_conv1d(ts_nb_filter)(inputs)
        if batchnorm:
            u = BatchNormalization()(u)
        u = Activation(activation)(u)
        if dropout_rate:
            u = Dropout(dropout_rate, name=name)(u)
    return u

def ts_upconv_unit(
    inputs,
    ts_nb_filter,
    concatenate_layer,
    name=None,
):
    # transposed convolution
    u = UpSampling1D(size=ts_upsize)(inputs)
    u = ts_conv1d(ts_nb_filter, ts_strides=None)(u)
    if batchnorm:
        u = BatchNormalization()(u)
    u = Activation(activation)(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    # u.shape TensorShape([None, 128, 18])
    # concatenate_layer.shape TensorShape([None, 126, 18])
    shape_diff = u.shape[1] - concatenate_layer.shape[1]
    if shape_diff > 0:
        crop_shape = (shape_diff // 2, shape_diff - shape_diff // 2)
    else:
        crop_shape = None

    if crop_shape:
        crop = Cropping1D(cropping=crop_shape)(u)
        upconv = concatenate([concatenate_layer, crop], name=name)
    elif not crop_shape:
        upconv = concatenate([concatenate_layer, u], name=name)

    return upconv

def ts_RRconv_unit(inputs, ts_nb_filter, ts_strides, name=None):

    if ts_strides == None:
        u = ts_conv_unit(inputs=inputs, ts_nb_filter=ts_nb_filter, ts_strides=None)
    else:
        u = ts_conv_unit(
            inputs=inputs, ts_nb_filter=ts_nb_filter, ts_strides=ts_strides
        )
    conv_1x1 = ts_conv1d(ts_nb_filter=ts_nb_filter, ts_strides=1)(u)
    for i in range(RRconv_time):
        if i == 0:
            r_u = u
        r_u = Add()([r_u, u])
        r_u = ts_conv_unit(inputs=r_u, ts_nb_filter=ts_nb_filter, ts_strides=None)

    return Add(name=name)([r_u, conv_1x1])

### spectrogram operation blocks
def spec_conv_block( inputs, spec_nb_filter, spec_kernel_size=(3, 3),
        kernel_regularizer=None,
        spec_strides=None, activation='relu', dropout=0.1, name=None):
    if (spec_strides != None) :
        u = Conv2D(filters=spec_nb_filter, kernel_size=spec_kernel_size,
             kernel_regularizer=kernel_regu,
            strides=spec_strides, padding='same')(inputs)
    else:
        u = Conv2D(filters=spec_nb_filter, kernel_size=spec_kernel_size,
             kernel_regularizer=kernel_regu,
            padding='same')(inputs)

    u = BatchNormalization()(u)
    u = Activation(activation)(u)
    u = Dropout(dropout, name=name)(u)
    return u

def spec_RRconv_block( inputs, spec_nb_filter, spec_kernel_size=(3, 3),
        kernel_regu=None,
        RRconv_time=2, spec_strides=None, activation='relu', 
        dropout=0.1, name=None):
    
    if spec_strides==None:
        u = spec_conv_block(inputs, spec_nb_filter=spec_nb_filter, 
        spec_kernel_size=spec_kernel_size, kernel_regu=kernel_regu,
        spec_strides=spec_strides, activation=activation, dropout=dropout)
    else:
        u = spec_conv_block(inputs, spec_nb_filter=spec_nb_filter, 
        spec_kernel_size=spec_kernel_size, kernel_regu=kernel_regu,
        spec_strides=spec_strides, activation=activation, dropout=dropout)

    conv_1x1 = Conv2D(spec_nb_filter, (1, 1))(u)
    for i in range(RRconv_time):
        if i == 0:
            r_u = u
        r_u = Add()([r_u, u])
        r_u = spec_conv_block(r_u, spec_nb_filter=spec_nb_filter, 
                spec_kernel_size=spec_kernel_size, 
                kernel_regu=kernel_regu,
                activation=activation, 
                dropout=dropout)
    return Add(name=name)([r_u, conv_1x1])


def spec_conv_block( inputs, spec_nb_filter, spec_kernel_size=(3, 3),
        kernel_regu=None,
        spec_strides=None, activation='relu', dropout=0.1, name=None):
    if (spec_strides != None) :
        u = Conv2D(filters=spec_nb_filter, kernel_size=spec_kernel_size,
             kernel_regularizer=kernel_regu,
            strides=spec_strides, padding='same')(inputs)
    else:
        u = Conv2D(filters=spec_nb_filter, kernel_size=spec_kernel_size,
             kernel_regularizer=kernel_regu,
            padding='same')(inputs)

    u = BatchNormalization()(u)
    u = Activation(activation)(u)
    u = Dropout(dropout, name=name)(u)
    return u

def spec_conv_unit(inputs, spec_nb_filter, spec_kernel_size=(3, 3),
         kernel_regu=None,
         spec_strides=(3, 3), activation='relu', dropout=0.1, name=None):

    u = spec_conv_block(inputs, spec_nb_filter=spec_nb_filter, 
            spec_kernel_size=spec_kernel_size,
            kernel_regu=kernel_regu,
            spec_strides=None, activation='relu', dropout=0.1)
    u = spec_conv_block(u, spec_nb_filter=spec_nb_filter, 
            spec_kernel_size=spec_kernel_size,
            kernel_regu=kernel_regu,
            spec_strides=spec_strides, activation='relu', dropout=0.1, name=name)
    return u

def spec_RRconv_unit(inputs, spec_nb_filter, spec_kernel_size=(3, 3),
         kernel_regu=None,
         spec_strides=(3, 3), activation='relu', dropout=0.1, name=None):

    u = spec_RRconv_block(inputs, spec_nb_filter=spec_nb_filter, 
            spec_kernel_size=spec_kernel_size,
            kernel_regu=kernel_regu,
            spec_strides=None, activation='relu', dropout=0.1)
    u = spec_RRconv_block(u, spec_nb_filter=spec_nb_filter, 
            spec_kernel_size=spec_kernel_size,
            kernel_regu=kernel_regu,
            spec_strides=spec_strides, activation='relu', dropout=0.1, name=name)
    return u

def spec_upRRconv_block(inputs, spec_nb_filter, spec_kernel_size=(3, 3),
        kernel_regu=None, spec_upsize=(1, 3),
        spec_strides=None, activation='relu', dropout=0.1):
    u = UpSampling2D(size=spec_upsize)(inputs)
    u = spec_RRconv_block( u, spec_nb_filter, 
            spec_kernel_size=spec_kernel_size, spec_strides=None, 
            kernel_regu=kernel_regu,
            activation=activation, dropout=dropout)
    u = BatchNormalization()(u)
    u = Activation(activation)(u)
    u = Dropout(dropout)(u)
    return u

def spec_upconv_block(inputs, spec_nb_filter, spec_kernel_size=(3, 3),
         kernel_regu=None, spec_upsize=(1, 3),
        spec_strides=None, activation='relu', dropout=0.1):
    u = UpSampling2D(size=spec_upsize)(inputs)
    u = spec_conv_block( u, spec_nb_filter, 
            spec_kernel_size=spec_kernel_size, spec_strides=None, 
            kernel_regu=kernel_regu,
            activation=activation, dropout=dropout)
    u = BatchNormalization()(u)
    u = Activation(activation)(u)
    u = Dropout(dropout)(u)
    return u

def spec_crop_concat(inputs, concatenate_layer, name=None):
    shape_diff1 = inputs.shape[1] - concatenate_layer.shape[1]
    shape_diff2 = inputs.shape[2] - concatenate_layer.shape[2]
    if shape_diff1 != 0 or shape_diff2!=0:
        crop_shape1 = (0, np.divmod(shape_diff1, 3)[1])                                                                                                                                                                                                                                                                                                                                                                                                                                  
        crop_shape2 = (np.divmod(shape_diff2, 3)[1], 0)
        crop_shape = (crop_shape1, crop_shape2)   
    else:
        crop_shape = None
    if crop_shape:
        crop = Cropping2D(cropping=crop_shape)(inputs)
        upconv = concatenate([concatenate_layer, crop], name=name)
    elif not crop_shape:
        upconv = concatenate([concatenate_layer, inputs], name=name)
    return upconv

def spec_upconv_unit(concatenate_layer=None, inputs=None, spec_nb_filter=None, 
        spec_kernel_size=(3, 3),  kernel_regu=None,
        spec_strides=None, activation='relu', spec_upsize=(1, 3),
        dropout=0.1, name=None):
    up_u = spec_upconv_block(inputs, spec_nb_filter, 
        spec_kernel_size=spec_kernel_size,
        kernel_regu=kernel_regu, spec_upsize=spec_upsize,
        spec_strides=spec_strides, activation=activation, dropout=dropout)
    concat_u = spec_crop_concat(up_u, concatenate_layer, name=name)
    return concat_u

def spec_upRRconv_unit(concatenate_layer=None, inputs=None, spec_nb_filter=None, 
        spec_kernel_size=(3, 3),  kernel_regu=None,
        spec_strides=None, activation='relu', spec_upsize=(1, 3),
        dropout=0.1, name=None):
    up_u = spec_upRRconv_block(inputs, spec_nb_filter, 
        spec_kernel_size=spec_kernel_size,
        kernel_regu=kernel_regu, spec_upsize=spec_upsize,
        spec_strides=spec_strides, activation=activation, dropout=dropout)
    concat_u = spec_crop_concat(up_u, concatenate_layer, name=name)
    return concat_u

def att_block(xl, gate, name=None):
    # xl = input feature (U net left hand side)
    # gate = gating signal (U net right hand side)
    F_l = int(xl.shape[1])
    F_g = int(gate.shape[1])
    F_int = int(xl.shape[2])

    W_x = Conv1D(F_l, F_int, strides=1, padding=padding,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu)(xl)
    W_x_n = BatchNormalization()(W_x)

    W_g = Conv1D(F_g, F_int, strides=1, padding=padding,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu)(gate)
    W_g_n = BatchNormalization()(W_g)

    add = Add()([W_x_n, W_g_n])
    add = Activation('relu')(add)

    psi = Conv1D(F_int, 1, strides=1, padding=padding,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu)(add)
    psi_n = BatchNormalization()(psi)
    psi_activate = Activation('sigmoid')(psi_n) 

    mul = Multiply(name=name)([xl, psi_activate])

    return mul

def fusion_unet(pretrained_weights=None):
    #################### build time-frequency model ####################
    # ========== Encoder
    spec_Es = []
    # initialize
    spec_inputs = Input(spec_input_size, name="spec_input")
    spec_conv_init = spec_conv_unit(
            inputs=spec_inputs,
            spec_nb_filter=spec_nb_filters[0],
            spec_kernel_size=spec_kernel_size,
            kernel_regu=kernel_regu,
            spec_strides=None,
            name=f"spec_E0",
        )


    spec_Es.append(spec_conv_init)

    # Encoder
    for i in range(len(spec_nb_filters) - 1):
        if i == 0:
            spec_E = spec_conv_unit(
                inputs=spec_conv_init,
                spec_nb_filter=spec_nb_filters[i+1],
                spec_kernel_size=spec_kernel_size,
                kernel_regu=kernel_regu,
                spec_strides=spec_strides,
                name=f"spec_E{i+1}",
            )
        else:
            spec_E = spec_conv_unit(
                inputs=spec_Es[-1],
                spec_nb_filter=spec_nb_filters[i+1],
                spec_kernel_size=spec_kernel_size,
                kernel_regu=kernel_regu,
                spec_strides=spec_strides,
                name=f"spec_E{i+1}",
            )       
        spec_Es.append(spec_E)

    #    #################### build time-series model ####################
    #    #spec_inputs = Input(spec_input_size, name="spec_input")


    # ========== Encoder
    ts_exp_Es = []
    ts_Es = []
    # initialize
    ts_inputs = Input(ts_input_size, name="ts_input")
    ts_conv_init_exp = ts_conv_unit(
        inputs=ts_inputs, ts_nb_filter=ts_nb_filters[0], 
        ts_strides=None, name="ts_E0"
    )

    ts_Es.append(ts_conv_init_exp)

    # Encoder
    for i in range(len(ts_nb_filters) - 1):
        if i == 0:
            ts_exp_E = ts_conv_unit(
                inputs=ts_conv_init_exp,
                ts_nb_filter=ts_nb_filters[i],
                ts_strides=None,
                name=f"ts_exp_E{i}",
            )

            ts_E = ts_conv_unit(
                inputs=ts_exp_E,
                ts_nb_filter=ts_nb_filters[i + 1],
                ts_strides=ts_strides,
                name=f"ts_E{i+1}",
            )
        else:
            ts_exp_E = ts_conv_unit(
                inputs=ts_E,
                ts_nb_filter=ts_nb_filters[i],
                ts_strides=None,
                name=f"ts_exp_E{i}",
            )

            ts_E = ts_conv_unit(
                inputs=ts_exp_E,
                ts_nb_filter=ts_nb_filters[i + 1],
                ts_strides=ts_strides,
                name=f"ts_E{i+1}",
            )
        ts_Es.append(ts_E)
        ts_exp_Es.append(ts_exp_E)

        # bottleneck layer
        if i == len(ts_nb_filters) - 2:
            ts_exp_E = ts_conv_unit(
                inputs=ts_E,
                ts_nb_filter=ts_nb_filters[i + 1],
                ts_strides=None,
                name=f"ts_exp_E{i+2}",
            )
            ts_exp_Es.append(ts_exp_E)

    # merge encoder bottleneck
    spec_transpose = tf.transpose(spec_Es[-1], [0, 3, 2, 1])
    spec_reshape = tf.squeeze(spec_transpose, axis=-1)
    spec_proj = tf.transpose(
        Dense(ts_exp_Es[-1].shape[1])(spec_reshape),
        [0, 2, 1]
    )
    feature_concat = concatenate([ts_exp_Es[-1], spec_proj])
    fusion = Conv1D(spec_proj.shape[-1], 1)(feature_concat)
    att = att_block(fusion,  ts_exp_Es[-1])
    trans = BatchNormalization(name=f"CrossFusion_init")(
        add([att, ts_exp_Es[-1]]))   

    # Decoder
    ts_Ds = []
    for i in range(len(ts_nb_filters)):
        if i == 0:
            ts_D = ts_upconv_unit(
                #inputs=ts_exp_Es[-1],
                inputs = trans,
                ts_nb_filter=ts_nb_filters[-1 - i],
                concatenate_layer=ts_exp_Es[-1 - i],
            )
        else:
            ts_D = ts_upconv_unit(
                inputs=ts_D_fus,
                ts_nb_filter=ts_nb_filters[-1 - i],
                concatenate_layer=ts_exp_Es[-1 - i],
            )

        ts_D_fus = ts_conv_unit(
            inputs=ts_D,
            ts_nb_filter=ts_nb_filters[-1 - i],
            ts_strides=None,
            name=f"ts_D{i}",
        )
        ts_Ds.append(ts_D_fus)

    ##========== Output map
    out_PS = Conv1D(
        3,
        1,
        kernel_initializer=kernel_init,
        kernel_regularizer=kernel_regu,
        name="pred_eqpick",
    )(ts_Ds[-1])

    out_EQM = Conv1D(
        2,
        1,
        kernel_initializer=kernel_init,
        kernel_regularizer=kernel_regu,
        name="pred_eqmask",
    )(ts_Ds[-1])

    out_eqpick_Act = Activation(out_activation, name="eq_picker")(out_PS)
    out_eqmask_Act = Activation(out_activation, name="eq_detector")(out_EQM)

    model = Model(inputs=[ts_inputs, spec_inputs], 
        outputs=[out_eqpick_Act, out_eqmask_Act])

    # compile
    if pretrained_weights==None:
        return model             

    else:
        model.load_weights(pretrained_weights)
        return model

def fusion_R2unet(pretrained_weights=None):
    #################### build time-frequency model ####################
    # ========== Encoder
    spec_Es = []
    # initialize
    spec_inputs = Input(spec_input_size, name="spec_input")
    spec_conv_init = spec_RRconv_unit(
            inputs=spec_inputs,
            spec_nb_filter=spec_nb_filters[0],
            spec_kernel_size=spec_kernel_size,
            kernel_regu=kernel_regu,
            spec_strides=None,
            name=f"spec_E0",
        )


    spec_Es.append(spec_conv_init)

    # Encoder
    for i in range(len(spec_nb_filters) - 1):
        if i == 0:
            spec_E = spec_RRconv_unit(
                inputs=spec_conv_init,
                spec_nb_filter=spec_nb_filters[i+1],
                spec_kernel_size=spec_kernel_size,
                kernel_regu=kernel_regu,
                spec_strides=spec_strides,
                name=f"spec_E{i+1}",
            )
        else:
            spec_E = spec_RRconv_unit(
                inputs=spec_Es[-1],
                spec_nb_filter=spec_nb_filters[i+1],
                spec_kernel_size=spec_kernel_size,
                kernel_regu=kernel_regu,
                spec_strides=spec_strides,
                name=f"spec_E{i+1}",
            )       
        spec_Es.append(spec_E)

    #    #################### build time-series model ####################
    #    #spec_inputs = Input(spec_input_size, name="spec_input")


    # ========== Encoder
    ts_exp_Es = []
    ts_Es = []
    # initialize
    ts_inputs = Input(ts_input_size, name="ts_input")
    ts_conv_init_exp = ts_RRconv_unit(
        inputs=ts_inputs, ts_nb_filter=ts_nb_filters[0], 
        ts_strides=None, name="ts_E0"
    )

    ts_Es.append(ts_conv_init_exp)

    # Encoder
    for i in range(len(ts_nb_filters) - 1):
        if i == 0:
            ts_exp_E = ts_RRconv_unit(
                inputs=ts_conv_init_exp,
                ts_nb_filter=ts_nb_filters[i],
                ts_strides=None,
                name=f"ts_exp_E{i}",
            )

            ts_E = ts_RRconv_unit(
                inputs=ts_exp_E,
                ts_nb_filter=ts_nb_filters[i + 1],
                ts_strides=ts_strides,
                name=f"ts_E{i+1}",
            )
        else:
            ts_exp_E = ts_RRconv_unit(
                inputs=ts_E,
                ts_nb_filter=ts_nb_filters[i],
                ts_strides=None,
                name=f"ts_exp_E{i}",
            )

            ts_E = ts_RRconv_unit(
                inputs=ts_exp_E,
                ts_nb_filter=ts_nb_filters[i + 1],
                ts_strides=ts_strides,
                name=f"ts_E{i+1}",
            )
        ts_Es.append(ts_E)
        ts_exp_Es.append(ts_exp_E)

        # bottleneck layer
        if i == len(ts_nb_filters) - 2:
            ts_exp_E = ts_RRconv_unit(
                inputs=ts_E,
                ts_nb_filter=ts_nb_filters[i + 1],
                ts_strides=None,
                name=f"ts_exp_E{i+2}",
            )
            ts_exp_Es.append(ts_exp_E)

    # merge encoder bottleneck
    spec_transpose = tf.transpose(spec_Es[-1], [0, 3, 2, 1])
    spec_reshape = tf.squeeze(spec_transpose, axis=-1)
    spec_proj = tf.transpose(
        Dense(ts_exp_Es[-1].shape[1])(spec_reshape),
        [0, 2, 1]
    )
    feature_concat = concatenate([ts_exp_Es[-1], spec_proj])
    fusion = Conv1D(spec_proj.shape[-1], 1)(feature_concat)
    att = att_block(fusion,  ts_exp_Es[-1])
    trans = BatchNormalization(name=f"CrossFusion_init")(
        add([att, ts_exp_Es[-1]]))   

    # Decoder
    ts_Ds = []
    for i in range(len(ts_nb_filters)):
        if i == 0:
            ts_D = ts_upconv_unit(
                #inputs=ts_exp_Es[-1],
                inputs = trans,
                ts_nb_filter=ts_nb_filters[-1 - i],
                concatenate_layer=ts_exp_Es[-1 - i],
            )
        else:
            ts_D = ts_upconv_unit(
                inputs=ts_D_fus,
                ts_nb_filter=ts_nb_filters[-1 - i],
                concatenate_layer=ts_exp_Es[-1 - i],
            )

        ts_D_fus = ts_RRconv_unit(
            inputs=ts_D,
            ts_nb_filter=ts_nb_filters[-1 - i],
            ts_strides=None,
            name=f"ts_D{i}",
        )
        ts_Ds.append(ts_D_fus)

    ##========== Output map
    out_PS = Conv1D(
        3,
        1,
        kernel_initializer=kernel_init,
        kernel_regularizer=kernel_regu,
        name="pred_eqpick",
    )(ts_Ds[-1])

    out_EQM = Conv1D(
        2,
        1,
        kernel_initializer=kernel_init,
        kernel_regularizer=kernel_regu,
        name="pred_eqmask",
    )(ts_Ds[-1])

    out_eqpick_Act = Activation(out_activation, name="eq_picker")(out_PS)
    out_eqmask_Act = Activation(out_activation, name="eq_detector")(out_EQM)

    model = Model(inputs=[ts_inputs, spec_inputs], 
        outputs=[out_eqpick_Act, out_eqmask_Act])

    # compile
    if pretrained_weights==None:
        return model             

    else:
        model.load_weights(pretrained_weights)
        return model