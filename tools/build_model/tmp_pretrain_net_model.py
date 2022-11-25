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

station_num = 4
wf_shape = (6000, 3)
spec_shape = (51, 601, 2)

wf_in_shape = [station_num, wf_shape[0], wf_shape[1]]
spec_in_shape = [station_num, 
    spec_shape[0], spec_shape[1], spec_shape[2]]
nb_filters = [6, 12, 18, 24, 30, 36]
kernel_init = 'he_uniform'
kernel_regu = None
activation = 'relu'
out_activation = 'softmax'
dropout_rate = 0.1
batchnorm = True
RRconv_time = 2
padding = 'same'
# time series parameters
ts_kernel_size = 7
ts_pool_size = 5
ts_strides = 5
ts_upsize = 5

# spectrogram parameters
spec_kernel_size = (3, 3)
spec_strides = (3, 3)
spec_upsize = 3

def conv1d( nb_filter, strides=None):
    if strides:
        return Conv1D(nb_filter, kernel_size=ts_kernel_size, 
            padding=padding, strides=strides,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu)
    else:
        return Conv1D(nb_filter, kernel_size=ts_kernel_size, 
            padding=padding,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu)

def conv_unit( inputs, nb_filter, strides, name=None):

    if (strides != None) :
        u = conv1d(nb_filter, strides=strides)(inputs)
        if batchnorm:
            u = BatchNormalization()(u)
        u = Activation(activation)(u)
        if dropout_rate:
            u = Dropout(dropout_rate, name=name)(u)
    
    else:
        u = conv1d(nb_filter)(inputs)
        if batchnorm:
            u = BatchNormalization()(u)
        u = Activation(activation)(u)
        if dropout_rate:
            u = Dropout(dropout_rate, name=name)(u)
    return u

def RRconv_unit( inputs, nb_filter, strides, name=None):
    
    if strides==None:
        u = conv_unit(inputs=inputs, 
            nb_filter=nb_filter, strides=None)
    else:
        u = conv_unit(inputs=inputs, 
            nb_filter=nb_filter, strides=strides)
    conv_1x1 = conv1d(nb_filter=nb_filter, strides=1)(u)
    for i in range(RRconv_time):
        if i == 0:
            r_u = u
        r_u = Add()([r_u, u])
        r_u = conv_unit(inputs=r_u, 
            nb_filter=nb_filter, strides=None)
    
    return Add(name=name)([r_u, conv_1x1])

def upconv_unit( inputs, nb_filter, concatenate_layer, 
        name=None):
        # transposed convolution
    u = UpSampling1D(size=ts_upsize)(inputs)
    u = conv1d(nb_filter, strides=None)(u)
    if batchnorm:
        u = BatchNormalization()(u)
    u = Activation(activation)(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    #u.shape TensorShape([None, 128, 18])
    # concatenate_layer.shape TensorShape([None, 126, 18])
    shape_diff = u.shape[1] - concatenate_layer.shape[1]
    if shape_diff > 0:
        crop_shape = (shape_diff//2, shape_diff-shape_diff//2)
    else:
        crop_shape = None

    if crop_shape:
        crop = Cropping1D(cropping=crop_shape)(u)      
        upconv = concatenate([concatenate_layer, crop], name=name)
    elif not crop_shape:
        upconv = concatenate([concatenate_layer, u], name=name)            

    return upconv

def spec_conv_block( inputs, spec_nb_filter, 
        spec_kernel_size=(3, 3), kernel_regu=None,
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

def spec_conv_unit( inputs, spec_nb_filter, 
        spec_kernel_size=(3, 3), kernel_regu=None,
        spec_strides=(3, 3), activation='relu', dropout=0.1, name=None):

    u = spec_conv_block(inputs, spec_nb_filter=spec_nb_filter, 
            spec_kernel_size=spec_kernel_size,
            kernel_regu=kernel_regu,
            spec_strides=None, activation='relu', dropout=0.1)
    u = spec_conv_block(u, spec_nb_filter=spec_nb_filter, 
            spec_kernel_size=spec_kernel_size,
            kernel_regu=kernel_regu,
            spec_strides=spec_strides, activation='relu', 
            dropout=0.1, name=name)
    return u

def spec_RRconv_block( inputs, spec_nb_filter, 
        spec_kernel_size=(3, 3),
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

def spec_RRconv_unit( inputs, spec_nb_filter, 
        spec_kernel_size=(3, 3),
        kernel_regu=None,
        spec_strides=(3, 3), activation='relu', dropout=0.1, name=None):

    u = spec_RRconv_block(inputs, spec_nb_filter=spec_nb_filter, 
            spec_kernel_size=spec_kernel_size,
            kernel_regu=kernel_regu,
            spec_strides=None, activation='relu', dropout=0.1)
    u = spec_RRconv_block(u, spec_nb_filter=spec_nb_filter, 
            spec_kernel_size=spec_kernel_size,
            kernel_regu=kernel_regu,
            spec_strides=spec_strides, activation='relu', 
            dropout=0.1, name=name)
    return u

def spec_upRRconv_block( inputs, spec_nb_filter, 
        spec_kernel_size=(3, 3),
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

def spec_upconv_block( inputs, spec_nb_filter, 
        spec_kernel_size=(3, 3),
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

def spec_crop_concat( inputs, concatenate_layer, name=None):
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

def spec_upconv_unit( concatenate_layer=None, inputs=None, 
        spec_nb_filter=None, 
        spec_kernel_size=(3, 3),  kernel_regu=None,
        spec_strides=None, activation='relu', spec_upsize=(1, 3),
        dropout=0.1, name=None):
    up_u = spec_upconv_block(inputs, spec_nb_filter, 
        spec_kernel_size=spec_kernel_size,
        kernel_regu=kernel_regu, spec_upsize=spec_upsize,
        spec_strides=spec_strides, activation=activation, dropout=dropout)
    concat_u = spec_crop_concat(up_u, concatenate_layer, name=name)
    return concat_u

def spec_upRRconv_unit( concatenate_layer=None, inputs=None, 
        spec_nb_filter=None, 
        spec_kernel_size=(3, 3),  kernel_regu=None,
        spec_strides=None, activation='relu', spec_upsize=(1, 3),
        dropout=0.1, name=None):
    up_u = spec_upRRconv_block(inputs, spec_nb_filter, 
        spec_kernel_size=spec_kernel_size,
        kernel_regu=kernel_regu, spec_upsize=spec_upsize,
        spec_strides=spec_strides, activation=activation, dropout=dropout)
    concat_u = spec_crop_concat(up_u, concatenate_layer, name=name)

    return concat_u

def spec_att_block( xl, gate, name=None):
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

def input_fusion(spec_in, ts_in, nb_filter):
    spec_up_factor = np.ceil(
        spec_in.shape[1]/ts_in.shape[2]).astype(int)
    spec_up = spec_upRRconv_block(ts_in, nb_filter, 
        spec_upsize=(1, spec_up_factor))
    spec_transpose = Dense(1)(tf.transpose(spec_up, [0, 2, 3, 1]))
    spec_reshape = tf.squeeze(spec_transpose, axis=-1)
    shape_difference = spec_reshape.shape[1] - spec_in.shape[1]
    spec_crop_shape = (shape_difference // 2, 
        shape_difference - shape_difference // 2)
    spec_crop = Cropping1D(cropping=spec_crop_shape)(spec_reshape)
    reduce_spec = Conv1D(spec_crop.shape[-1], 1)(spec_crop)

    data_concat = concatenate([reduce_spec, spec_in])
    fusion = Conv1D(spec_crop.shape[-1], 1)(data_concat)
    #att = MultiHeadAttention(num_heads=8, key_dim=ts_nb_filters[0])(
    #    fusion,  spec_in)
    #trans = BatchNormalization(name=f"CrossFusion_init")(
    #    add([att, spec_in]))    
    return fusion

def R2unet_fusion_tmp(pretrained_weights=None):

    wf_in = Input(wf_in_shape)
    wf_sta = [wf_in[:, I, ...] for I in range(station_num)]

    spec_in = Input(spec_in_shape)
    wf_spec = [spec_in[:, I, ...] for I in range(station_num)]
    
    ## merge the input
    # initialize
    wf_sta_init = []
    spec_sta_init = []
    for W in range(station_num):
        ts_init = RRconv_unit(
            inputs=wf_sta[W], nb_filter=nb_filters[0], 
            strides=None, name=f"ts_init_sta{W:02}"
        )
        spec_init = spec_RRconv_unit(
                inputs=wf_spec[W],
                spec_nb_filter=nb_filters[0],
                spec_kernel_size=spec_kernel_size,
                kernel_regu=kernel_regu,
                spec_strides=None,
                name=f"spec_init_sta{W:02}",
        )
        wf_sta_init.append(ts_init)
        spec_sta_init.append(spec_init)
    # merge
    merge_input_sta = []
    for W1 in range(station_num):
        merge_input = input_fusion(
            wf_sta_init[W1], spec_sta_init[W1], nb_filters[0])
        merge_input_sta.append(merge_input)

    sta_Es = []
    for sid in range(station_num):
        # initial
        #========== Encoder
        exp_Es = []
        Es = []    

        # initialize
        conv_init_exp = RRconv_unit(inputs=merge_input_sta[sid],
            nb_filter=nb_filters[0], strides=None, 
            name=f'E0_sta{sid:02}')

        Es.append(conv_init_exp)

        # Encoder
        for i in range(len(nb_filters)-1):
            if i == 0:
                exp_E = RRconv_unit(inputs=conv_init_exp, 
                    nb_filter=nb_filters[i],
                    strides=None, name=f'exp_E{i}_sta{sid:02}') 

                E = RRconv_unit(inputs=exp_E, 
                    nb_filter=nb_filters[i+1], 
                    strides=ts_strides,
                    name=f'E{i+1}_sta{sid:02}')
            else:
                exp_E = RRconv_unit(inputs=E, 
                    nb_filter=nb_filters[i], strides=None, 
                    name=f'exp_E{i}_sta{sid:02}')  

                E = RRconv_unit(inputs=exp_E, 
                    nb_filter=nb_filters[i+1], 
                    strides=ts_strides, name=f'E{i+1}_sta{sid:02}') 
            Es.append(E)
            exp_Es.append(exp_E)

            # bottleneck layer
            if i == len(nb_filters)-2:
                exp_E = RRconv_unit(inputs=E, 
                    nb_filter=nb_filters[i+1], strides=None, 
                    name=f'exp_E{i+2}_sta{sid:02}')
                exp_Es.append(exp_E)

        sta_Es.append(exp_Es)

    # merge the encoders
    merge_Es = []
    for s in range(len(nb_filters)):
        # merge the encoder features
        stack_enc = tf.transpose(
            tf.stack([E[s] for E in sta_Es], axis=1),
            [0, 2, 3, 1]
        )
        merge_enc = tf.squeeze(Conv1D(1, 1)(stack_enc), axis=-1)
        merge_Es.append(merge_enc)
    
    # Decoder
    sta_Ds = []
    for sid in range(station_num):
        Ds = []
        for j in range(len(nb_filters)):
            if j == 0:     
                D = upconv_unit(inputs=merge_Es[-1], 
                    nb_filter=nb_filters[-1-j], 
                    concatenate_layer=sta_Es[sid][-1-j])

            else:
                D = upconv_unit(inputs=D_fus, 
                    nb_filter=nb_filters[-1-j], 
                    concatenate_layer=sta_Es[sid][-1-j]) 
                            
            D_fus = RRconv_unit(inputs=D, 
                    nb_filter=nb_filters[-1-j], 
                    strides=None, name=f'D{j}_merge_sta{sid:02}')

            Ds.append(D_fus)
        sta_Ds.append(Ds)

    merge_D = tf.squeeze(Conv1D(1, 1)(
        tf.stack([DD[-1] for DD in sta_Ds], axis=-1)), axis=-1
    )
    
    out_EQPS_collect = []
    out_EQM_collect = []
    out_RFM_collect = []
    for S in range(station_num):
        ##========== Output map
        outPS = Conv1D(3, 1,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu,
            name=f'pred_PS_sta{S:02}')(sta_Ds[S][-1])
        outM = Conv1D(2, 1,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu,
            name=f'pred_eqM_sta{S:02}')(sta_Ds[S][-1])
        outM_RF = Conv1D(2, 1,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu,
            name=f'pred_rfM_sta{S:02}')(sta_Ds[S][-1])            
        out_EQPS_collect.append(outPS)
        out_EQM_collect.append(outM)
        out_RFM_collect.append(outM_RF)

    out_sta_EQpick = tf.stack(out_EQPS_collect, axis=1)
    out_sta_EQmask = tf.stack(out_EQM_collect, axis=1)
    out_sta_RFmask = tf.stack(out_RFM_collect, axis=1)

    outEQocc = Conv1D(2, 1,
        kernel_initializer=kernel_init, 
        kernel_regularizer=kernel_regu,
        name=f'pred_EQocc')(merge_D)
    outRFocc = Conv1D(2, 1,
        kernel_initializer=kernel_init, 
        kernel_regularizer=kernel_regu,
        name=f'pred_RFocc')(merge_D)        

    outPS_Act = Activation(
        out_activation, name=f'sta_eq_picker')(out_sta_EQpick)
    outM_Act = Activation(
        out_activation, name=f'sta_eq_detector')(out_sta_EQmask)
    outM_RF_Act = Activation(
        out_activation, name=f'sta_rf_detector')(out_sta_RFmask)
    out_eqocc = Activation(out_activation, name=f'eqocc')(outEQocc)
    out_rfocc = Activation(out_activation, name=f'rfocc')(outRFocc)

    model = Model(inputs=[wf_in, spec_in], 
        outputs=[outPS_Act, outM_Act, outM_RF_Act, out_eqocc, out_rfocc])

    # compile
    if pretrained_weights==None:
        return model             

    else:
        model.load_weights(pretrained_weights)    
        return model       
        
def R2unet_fusion(pretrained_weights=None):

    wf_in = Input(wf_in_shape)
    wf_sta = [wf_in[:, I, ...] for I in range(station_num)]

    spec_in = Input(spec_in_shape)
    wf_spec = [spec_in[:, I, ...] for I in range(station_num)]

    sta_spec_Es = []
    for sid in range(station_num):
        spec_Es = []
        # initialize
        spec_conv_init = spec_RRconv_unit(
                inputs=wf_spec[sid],
                spec_nb_filter=nb_filters[0],
                spec_kernel_size=spec_kernel_size,
                kernel_regu=kernel_regu,
                spec_strides=None,
                name=f"spec_E0_sta{sid:02}"
            )

        spec_Es.append(spec_conv_init)

        # Encoder
        for i in range(len(nb_filters) - 1):
            if i == 0:
                spec_E = spec_RRconv_unit(
                    inputs=spec_conv_init,
                    spec_nb_filter=nb_filters[i+1],
                    spec_kernel_size=spec_kernel_size,
                    kernel_regu=kernel_regu,
                    spec_strides=spec_strides,
                    name=f"spec_E{i+1}_sta{sid:02}",
                )
            else:
                spec_E = spec_RRconv_unit(
                    inputs=spec_Es[-1],
                    spec_nb_filter=nb_filters[i+1],
                    spec_kernel_size=spec_kernel_size,
                    kernel_regu=kernel_regu,
                    spec_strides=spec_strides,
                    name=f"spec_E{i+1}_sta{sid:02}",
                )       
            spec_Es.append(spec_E)
        sta_spec_Es.append(spec_Es)

    ## time series data

    sta_Es = []
    for sid in range(station_num):
        # initial
        #========== Encoder
        exp_Es = []
        Es = []    

        # initialize
        conv_init_exp = RRconv_unit(inputs=wf_sta[sid],
            nb_filter=nb_filters[0], strides=None, 
            name=f'E0_sta{sid:02}')

        Es.append(conv_init_exp)

        # Encoder
        for i in range(len(nb_filters)-1):
            if i == 0:
                exp_E = RRconv_unit(inputs=conv_init_exp, 
                    nb_filter=nb_filters[i],
                    strides=None, name=f'exp_E{i}_sta{sid:02}') 

                E = RRconv_unit(inputs=exp_E, 
                    nb_filter=nb_filters[i+1], 
                    strides=ts_strides,
                    name=f'E{i+1}_sta{sid:02}')
            else:
                exp_E = RRconv_unit(inputs=E, 
                    nb_filter=nb_filters[i], strides=None, 
                    name=f'exp_E{i}_sta{sid:02}')  

                E = RRconv_unit(inputs=exp_E, 
                    nb_filter=nb_filters[i+1], 
                    strides=ts_strides, name=f'E{i+1}_sta{sid:02}') 
            Es.append(E)
            exp_Es.append(exp_E)

            # bottleneck layer
            if i == len(nb_filters)-2:
                exp_E = RRconv_unit(inputs=E, 
                    nb_filter=nb_filters[i+1], strides=None, 
                    name=f'exp_E{i+2}_sta{sid:02}')
                exp_Es.append(exp_E)

        sta_Es.append(exp_Es)

    merge_Es = []
    for s in range(len(nb_filters)):
        # merge the encoder features
        stack_enc = tf.transpose(
            tf.stack([E[s] for E in sta_Es], axis=1),
            [0, 2, 3, 1]
        )
        merge_enc = tf.squeeze(Conv1D(1, 1)(stack_enc), axis=-1)
        merge_Es.append(merge_enc)

    # fuse the time-series features and time-frequency features
    sta_fuse_enc = []
    for sn in range(station_num):
        spec_transpose = tf.transpose(sta_spec_Es[sn][-1], [0, 3, 2, 1])
        spec_reshape = tf.squeeze(spec_transpose, axis=-1)
        spec_proj = tf.transpose(
            Dense(merge_Es[-1].shape[1])(spec_reshape),
            [0, 2, 1]
        )
        feature_concat = concatenate([merge_Es[-1], spec_proj])
        fusion = Conv1D(spec_proj.shape[-1], 1)(feature_concat)
        att = spec_att_block(fusion,  merge_Es[-1])
        trans = BatchNormalization(name=f"CrossFusion_init_sta{sn}")(
            add([att, merge_Es[-1]]))
        sta_fuse_enc.append(trans)
    
    # Decoder
    sta_Ds = []
    for sid in range(station_num):
        Ds = []
        for j in range(len(nb_filters)):
            if j == 0:     
                D = upconv_unit(inputs=sta_fuse_enc[sid], 
                    nb_filter=nb_filters[-1-j], 
                    concatenate_layer=sta_Es[sid][-1-j])

            else:
                D = upconv_unit(inputs=D_fus, 
                    nb_filter=nb_filters[-1-j], 
                    concatenate_layer=sta_Es[sid][-1-j]) 
                            
            D_fus = RRconv_unit(inputs=D, 
                    nb_filter=nb_filters[-1-j], 
                    strides=None, name=f'D{j}_merge_sta{sid:02}')

            Ds.append(D_fus)
        sta_Ds.append(Ds)
    merge_D = tf.squeeze(Conv1D(1, 1)(
        tf.stack([DD[-1] for DD in sta_Ds], axis=-1)), axis=-1
    )
    
    out_EQPS_collect = []
    out_EQM_collect = []
    out_RFM_collect = []
    for S in range(station_num):
        ##========== Output map
        outPS = Conv1D(3, 1,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu,
            name=f'pred_PS_sta{S:02}')(sta_Ds[S][-1])
        outM = Conv1D(2, 1,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu,
            name=f'pred_M_sta{S:02}')(sta_Ds[S][-1])
        outM_RF = Conv1D(2, 1,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu,
            name=f'pred_M_sta{S:02}')(sta_Ds[S][-1])            
        out_EQPS_collect.append(outPS)
        out_EQM_collect.append(outM)
        out_RFM_collect.append(outM_RF)

    out_sta_EQpick = tf.stack(out_EQPS_collect, axis=1)
    out_sta_EQmask = tf.stack(out_EQM_collect, axis=1)
    out_sta_RFmask = tf.stack(out_RFM_collect, axis=1)

    outEQocc = Conv1D(2, 1,
        kernel_initializer=kernel_init, 
        kernel_regularizer=kernel_regu,
        name=f'pred_EQocc')(merge_D)
    outRFocc = Conv1D(2, 1,
        kernel_initializer=kernel_init, 
        kernel_regularizer=kernel_regu,
        name=f'pred_RFocc')(merge_D)        


    outPS_Act = Activation(
        out_activation, name=f'sta_eq_picker')(out_sta_EQpick)
    outM_Act = Activation(
        out_activation, name=f'sta_eq_detector')(out_sta_EQmask)
    outM_RF_Act = Activation(
        out_activation, name=f'sta_rf_detector')(out_sta_RFmask)
    out_eqocc = Activation(out_activation, name=f'eqocc')(outEQocc)
    out_rfocc = Activation(out_activation, name=f'rfocc')(outRFocc)

    model = Model(inputs=[wf_in, spec_in], 
        outputs=[outPS_Act, outM_Act, outM_RF_Act, out_eqocc, out_rfocc])
    # compile
    if pretrained_weights==None:
        return model             

    else:
        model.load_weights(pretrained_weights)    
        return model        


def R2unet_wf(pretrained_weights=None):
    wf_in = Input(wf_in_shape)
    wf_sta = [wf_in[:, I, ...] for I in range(station_num)]

    sta_Es = []
    for sid in range(station_num):
        # initial
        #========== Encoder
        exp_Es = []
        Es = []    

        # initialize
        conv_init_exp = RRconv_unit(inputs=wf_sta[sid],
            nb_filter=nb_filters[0], strides=None, 
            name=f'E0_sta{sid:02}')

        Es.append(conv_init_exp)

        # Encoder
        for i in range(len(nb_filters)-1):
            if i == 0:
                exp_E = RRconv_unit(inputs=conv_init_exp, 
                    nb_filter=nb_filters[i],
                    strides=None, name=f'exp_E{i}_sta{sid:02}') 

                E = RRconv_unit(inputs=exp_E, 
                    nb_filter=nb_filters[i+1], 
                    strides=ts_strides,
                    name=f'E{i+1}_sta{sid:02}')
            else:
                exp_E = RRconv_unit(inputs=E, 
                    nb_filter=nb_filters[i], strides=None, 
                    name=f'exp_E{i}_sta{sid:02}')  

                E = RRconv_unit(inputs=exp_E, 
                    nb_filter=nb_filters[i+1], 
                    strides=ts_strides, name=f'E{i+1}_sta{sid:02}') 
            Es.append(E)
            exp_Es.append(exp_E)

            # bottleneck layer
            if i == len(nb_filters)-2:
                exp_E = RRconv_unit(inputs=E, 
                    nb_filter=nb_filters[i+1], strides=None, 
                    name=f'exp_E{i+2}_sta{sid:02}')
                exp_Es.append(exp_E)

        sta_Es.append(exp_Es)

    # Decoder
    sta_Ds = []
    for sid in range(station_num):
        Ds = []
        for j in range(len(nb_filters)):
            if j == 0:     
                D = upconv_unit(inputs=sta_Es[sid][-1], 
                    nb_filter=nb_filters[-1-j], 
                    concatenate_layer=sta_Es[sid][-1-j])

            else:
                D = upconv_unit(inputs=D_fus, 
                    nb_filter=nb_filters[-1-j], 
                    concatenate_layer=sta_Es[sid][-1-j]) 
                            
            D_fus = RRconv_unit(inputs=D, 
                    nb_filter=nb_filters[-1-j], 
                    strides=None, name=f'D{j}_merge_sta{sid:02}')

            Ds.append(D_fus)
        sta_Ds.append(Ds)

    merge_D = tf.squeeze(Conv1D(1, 1)(
        tf.stack([DD[-1] for DD in sta_Ds], axis=-1)), axis=-1
    )
    out_EQPS_collect = []
    out_EQM_collect = []
    for S in range(station_num):
        ##========== Output map
        outPS = Conv1D(3, 1,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu,
            name=f'pred_PS_sta{S:02}')(sta_Ds[S][-1])
        outM = Conv1D(2, 1,
            kernel_initializer=kernel_init, 
            kernel_regularizer=kernel_regu,
            name=f'pred_M_sta{S:02}')(sta_Ds[S][-1])
        out_EQPS_collect.append(outPS)
        out_EQM_collect.append(outM)

    out_sta_EQpick = tf.stack(out_EQPS_collect, axis=1)
    out_sta_EQmask = tf.stack(out_EQM_collect, axis=1)
    outEQocc = Conv1D(2, 1,
        kernel_initializer=kernel_init, 
        kernel_regularizer=kernel_regu,
        name=f'pred_EQocc')(merge_D)

    outPS_Act = Activation(out_activation, name=f'sta_picker')(out_sta_EQpick)
    outM_Act = Activation(out_activation, name=f'sta_detector')(out_sta_EQmask)
    out_eqocc = Activation(out_activation, name=f'eqocc')(outEQocc)

    model = Model(inputs=wf_in, 
        outputs=[outPS_Act, outM_Act, out_eqocc])

    # compile
    if pretrained_weights==None:
        return model             

    else:
        model.load_weights(pretrained_weights)    
        return model