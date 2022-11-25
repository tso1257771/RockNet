from rocknet_1sta_model import fusion_unet
from model_utils import *

def associate_net(
        single_sta_model_h5=None, 
        station_num=4, 
        pretrained_weights=None
):
    
    # make inference on input data
    wf_in = Input([station_num, 6000, 3])
    spec_in = Input([ station_num, 51, 601, 2])
    
    wf_sta = [wf_in[:, I, ...] for I in range(station_num)]
    wf_spec = [spec_in[:, I, ...] for I in range(station_num)]
    batch_size = tf.shape(wf_in)[0]

    wf_input = tf.reshape(
        wf_in,
        [batch_size*station_num, 6000, 3]
    )

    spec_input = tf.reshape(
        spec_in,
        [batch_size*station_num, 51, 601, 2]
    )


    ### single-station predictive model
    if single_sta_model_h5:
        pre_single_model = fusion_unet(
            single_sta_model_h5)
    else:
        pre_single_model = fusion_unet()

    # freeze all layers despite the predictive ones
    layer_name = np.array([n.name for n in pre_single_model.layers])
    last_D_idx = np.where(layer_name=='ts_D4')[0][0]
    for s_layers in pre_single_model.layers:#[:last_D_idx+1]:
        s_layers.trainable = False
    #last_D_block = pre_single_model.get_layer('ts_D4').output

    enc_block  = [
        pre_single_model.get_layer(f'ts_E{I}').output for I in range(
            len(ts_nb_filters)) 
    ]
    fusion_block = pre_single_model.get_layer('CrossFusion_init').output


    detect_model = Model(
        inputs=pre_single_model.input,
        outputs=[
            enc_block,
            fusion_block,
            pre_single_model.output
        ]
    )
    _enc_block, _fusion_block, [_eqpick, _eqmask, _rfmask] = \
        detect_model([wf_input, spec_input])

    # reshape the model output for association model input
    merge_enc = []
    for e in range(len(_enc_block)):
        enc_shape = _enc_block[e].shape[1:]
        _enc = tf.reshape(_enc_block[e],
            [batch_size, station_num, enc_shape[0], enc_shape[1]], 
            name=f'enc_stations_{e}'
        )
        merge_enc_tensor = tf.squeeze(Conv1D(1, 1)(
            tf.transpose(_enc, [0, 2, 3, 1])), axis=-1)
        merge_enc.append(merge_enc_tensor)

    fusion_block = tf.squeeze(
        Conv1D(1, 1)(
            tf.transpose(
                tf.reshape(
                    _fusion_block,
                    [batch_size, station_num, 10, 30]
                ), [0, 2, 3, 1]
            )
        ), axis=-1, name='fusion_block'
    )

    # Decoder
    Ds = []
    for j in range(len(ts_nb_filters)):
        if j == 0:     
            D = ts_upconv_unit(inputs=fusion_block, 
                ts_nb_filter=ts_nb_filters[-1-j], 
                concatenate_layer=merge_enc[-1-j]
            )
            #    ts_upsize=5)

        else:
            D = ts_upconv_unit(inputs=D_fus, 
                ts_nb_filter=ts_nb_filters[-1-j], 
                concatenate_layer=merge_enc[-1-j]
            )
            #    ts_upsize=5)
                        
        D_fus = ts_RRconv_unit(inputs=D, 
                ts_nb_filter=ts_nb_filters[-1-j],
                RRconv_time=3,
                ts_strides=None, name=f'D{j}_net')

        Ds.append(D_fus)

    # single-model direct output
    eqpick = tf.reshape(
        _eqpick,
        [batch_size, station_num, 6000, 3], name='eqpick_output'
    )
    eqmask = tf.reshape(
        _eqmask,
        [batch_size, station_num, 6000, 2], name='eqmask_output'
    )
    rfmask = tf.reshape(
        _rfmask,
        [batch_size, station_num, 6000, 2], name='rfmask_output'
    )


    rf_occ = BatchNormalization(name='rfocc')(
        Conv1D(2, 1, name='BiLSTM_rfocc')(
            Bidirectional(LSTM(
                2, return_sequences=True, dropout=0.1))(Ds[-1])
        )
    )

    eq_occ = BatchNormalization(name='eqocc')(
        Conv1D(2, 1, name='BiLSTM_eqocc')(
            Bidirectional(LSTM(
                2, return_sequences=True, dropout=0.1))(Ds[-1])
        )
    )
    eq_Act = Activation('softmax', name="eq_occ_act")(eq_occ)
    rf_Act = Activation('softmax', name="rf_occ_act")(rf_occ)


    net_model = Model(inputs=[wf_in, spec_in], 
        outputs=[rf_Act, eq_Act, eqpick, eqmask, rfmask])
    # compile
    if pretrained_weights==None:
        return net_model             

    else:
        net_model.load_weights(pretrained_weights)
        return net_model