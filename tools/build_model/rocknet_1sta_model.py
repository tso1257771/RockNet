from model_utils import *

def fusion_unet(pretrained_weights=None):
    #################### build spectrogram encoder #########
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

    ##################### build waveform encoder ####################
    # ========== Encoder
    ts_exp_Es = []
    ts_Es = []
    # initialize
    ts_inputs = Input(ts_input_size, name="ts_input")
    ts_conv_init_exp = ts_RRconv_unit(
        inputs=ts_inputs, ts_nb_filter=ts_nb_filters[0], 
        ts_strides=None, RRconv_time=RRconv_time,
        name="ts_E0"
    )
    ts_Es.append(ts_conv_init_exp)

    # Encoder
    for i in range(len(ts_nb_filters) - 1):
        if i == 0:
            ts_exp_E = ts_RRconv_unit(
                inputs=ts_conv_init_exp,
                ts_nb_filter=ts_nb_filters[i],
                ts_strides=None,
                RRconv_time=RRconv_time,
                name=f"ts_exp_E{i}",
            )

            ts_E = ts_RRconv_unit(
                inputs=ts_exp_E,
                ts_nb_filter=ts_nb_filters[i + 1],
                ts_strides=ts_strides,
                RRconv_time=RRconv_time,
                name=f"ts_E{i+1}",
            )
        else:
            ts_exp_E = ts_RRconv_unit(
                inputs=ts_E,
                ts_nb_filter=ts_nb_filters[i],
                ts_strides=None,
                RRconv_time=RRconv_time,
                name=f"ts_exp_E{i}",
            )

            ts_E = ts_RRconv_unit(
                inputs=ts_exp_E,
                ts_nb_filter=ts_nb_filters[i + 1],
                ts_strides=ts_strides,
                RRconv_time=RRconv_time,
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
                RRconv_time=RRconv_time,
                name=f"ts_exp_E{i+2}",
            )
            ts_exp_Es.append(ts_exp_E)

    # feature fusion block
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
            RRconv_time=RRconv_time,
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

    out_RFM = Conv1D(
        2,
        1,
        kernel_initializer=kernel_init,
        kernel_regularizer=kernel_regu,
        name="pred_rfmask",
    )(ts_Ds[-1])

    out_eqpick_Act = Activation(
        out_activation, name="eq_picker")(out_PS)
    out_eqmask_Act = Activation(
        out_activation, name="eq_detector")(out_EQM)
    out_rfmask_Act = Activation(
        out_activation, name="rf_detector")(out_RFM)

    model = Model(inputs=[ts_inputs, spec_inputs], 
        outputs=[out_eqpick_Act, out_eqmask_Act, out_rfmask_Act])

    # compile
    if pretrained_weights==None:
        return model             

    else:
        model.load_weights(pretrained_weights)
        return model

def wf_unet(pretrained_weights=None):
    ##################### build waveform encoder ####################
    # ========== Encoder
    ts_exp_Es = []
    ts_Es = []
    # initialize
    ts_inputs = Input(ts_input_size, name="ts_input")
    ts_conv_init_exp = ts_RRconv_unit(
        inputs=ts_inputs, ts_nb_filter=ts_nb_filters[0], 
        ts_strides=None, RRconv_time=RRconv_time,
        name="ts_E0"
    )
    ts_Es.append(ts_conv_init_exp)

    # Encoder
    for i in range(len(ts_nb_filters) - 1):
        if i == 0:
            ts_exp_E = ts_RRconv_unit(
                inputs=ts_conv_init_exp,
                ts_nb_filter=ts_nb_filters[i],
                ts_strides=None,
                RRconv_time=RRconv_time,
                name=f"ts_exp_E{i}",
            )

            ts_E = ts_RRconv_unit(
                inputs=ts_exp_E,
                ts_nb_filter=ts_nb_filters[i + 1],
                ts_strides=ts_strides,
                RRconv_time=RRconv_time,
                name=f"ts_E{i+1}",
            )
        else:
            ts_exp_E = ts_RRconv_unit(
                inputs=ts_E,
                ts_nb_filter=ts_nb_filters[i],
                ts_strides=None,
                RRconv_time=RRconv_time,
                name=f"ts_exp_E{i}",
            )

            ts_E = ts_RRconv_unit(
                inputs=ts_exp_E,
                ts_nb_filter=ts_nb_filters[i + 1],
                ts_strides=ts_strides,
                RRconv_time=RRconv_time,
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
                RRconv_time=RRconv_time,
                name=f"ts_exp_E{i+2}",
            )
            ts_exp_Es.append(ts_exp_E)

    # Decoder
    ts_Ds = []
    for i in range(len(ts_nb_filters)):
        if i == 0:
            ts_D = ts_upconv_unit(
                #inputs=ts_exp_Es[-1],
                inputs = ts_exp_Es[-1 - i],
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
            RRconv_time=RRconv_time,
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

    out_RFM = Conv1D(
        2,
        1,
        kernel_initializer=kernel_init,
        kernel_regularizer=kernel_regu,
        name="pred_rfmask",
    )(ts_Ds[-1])

    out_eqpick_Act = Activation(
        out_activation, name="eq_picker")(out_PS)
    out_eqmask_Act = Activation(
        out_activation, name="eq_detector")(out_EQM)
    out_rfmask_Act = Activation(
        out_activation, name="rf_detector")(out_RFM)

    model = Model(inputs=ts_inputs, 
        outputs=[out_eqpick_Act, out_eqmask_Act, out_rfmask_Act])

    # compile
    if pretrained_weights==None:
        return model             

    else:
        model.load_weights(pretrained_weights)
        return model