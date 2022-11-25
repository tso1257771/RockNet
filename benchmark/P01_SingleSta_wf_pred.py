import os
import sys
sys.path.append('../')
sys.path.append('../tools/build_model')
import logging
import shutil
import numpy as np
import scipy.signal as ss
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_addons.optimizers import RectifiedAdam
from glob import glob
from scipy.signal import find_peaks
from obspy.signal.trigger import trigger_onset
from tools.rockfall_STMF_fusion import compute_STFT
from tools.RF_example_parser import tfrecord_dataset_fusion_RF
from tools.EQ_picker import picker
from tools.data_utils import pick_peaks
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
logging.basicConfig(level=logging.INFO,
    format='%(levelname)s : %(asctime)s : %(message)s')

basepath = '../'
datapath = '../data/tfrecord_SingleSta'
mdl_hdr = 'wf_RF_60s'
RF_model = f'../trained_model/{mdl_hdr}/train.hdf5'

## load model using only waveform as input
model = tf.keras.models.load_model(RF_model, 
    custom_objects={
    'optimizer':RectifiedAdam(learning_rate=1e-4) , 
    'loss':tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE), 
    'metrics':['loss']
})
outpath = os.path.join('./pred_results', mdl_hdr)
if os.path.exists(outpath):
    shutil.rmtree(outpath)
os.makedirs(outpath)

plt_figure = False
tfr_dir = [
    os.path.join(datapath, 'RF_multiple_sta_label/test'),
    os.path.join(datapath, 'car_multiple_sta_label/test'),
    os.path.join(datapath, 'engineering/test'),
    os.path.join(datapath, 'EQ/test')
]
tfr_lbl = [
    'rockfall', 'nz_car', 'nz_engineering', 'earthquake'
]
picker_args = {
    'P_threshold':0.3, 
    'S_threshold':0.3, 
    'detection_threshold':0.5
}
for T in range(len(tfr_dir)):
    label_type = tfr_lbl[T]
    tfr_glob = np.sort(glob(os.path.join(tfr_dir[T], '*.tfrecord')))

    message_out = os.path.join(outpath, f'SingSta_pred_{label_type}.txt')
    figout = os.path.join(outpath, f'fig_SingSta_pred_{label_type}')
    if plt_figure:
        os.makedirs(figout)

    if label_type != 'earthquake':
        message_hdr = 'evid, label_type, pred_type, eq_prob, rf_prob\n'
    else:
        message_hdr = '        evid,   sta, chn,  |manP-manS|'+\
            ', manP,  manS, predP,'+\
            ' predP_prob, predS, predS_prob,'+\
            '       dist, hp_Psnr, hp_Ssnr, real_type,'+\
            ' pred_type, eq_prob, rf_prob\n'
    f_out = open(message_out, 'w')
    f_out.write(message_hdr)

    ct = 0
    split_iter = 2
    sep_idx = int(len(tfr_glob)//split_iter)
    for i in range(split_iter):
        st_idx, end_idx = sep_idx*i, sep_idx*(i+1)
        if i != split_iter-1:
            pred_batch_list = tfr_glob[st_idx:end_idx]
            n_pred = len(pred_batch_list)
            pred_batch = iter(tfrecord_dataset_fusion_RF(
                pred_batch_list,
                repeat=1, data_length=6000, 
                batch_size=n_pred, shuffle_buffer_size=None))
        else:
            pred_batch_list = tfr_glob[st_idx:]
            n_pred = len(pred_batch_list)
            pred_batch = iter(tfrecord_dataset_fusion_RF(
                pred_batch_list, 
                repeat=1, data_length=6000,
                batch_size=n_pred, shuffle_buffer_size=None))
                
        trc, spec, EQ_pick, EQ_mask, RF_mask, idx = next(pred_batch)
        pred_pick, pred_eq, pred_RF = model.predict(trc)

        for k in range(n_pred):
            logging.info(f"Generating predictions ... "+\
                f"{ct+1}/{len(tfr_glob)}: "+\
                f"{idx[k].numpy().decode('utf-8')}")
            evid = idx[k].numpy().decode('utf-8')
            sta_info = idx[k].numpy().decode('utf-8').split('_')[1]

            pred_eq_p = pred_pick[k].T[0]
            pred_eq_s = pred_pick[k].T[1]
            pred_eq_mask = pred_eq[k].T[0]
            pred_rf_mask = pred_RF[k].T[0]
            gt_mask = RF_mask[k].numpy().T[0]

            trigger_RFmask = trigger_onset(pred_rf_mask, 0.5, 0.5)
            trigger_EQmask = trigger_onset(pred_eq_mask, 0.5, 0.5)

            ### ground-truth mask vs predictions
            if label_type == 'rockfall':
                gt_rf_st, gt_rf_end = trigger_onset(gt_mask, 0.1, 0.1)[0]

                trigger_RFmask_pb = np.mean(pred_rf_mask[gt_rf_st:gt_rf_end])
                trigger_EQmask_pb = np.mean(pred_eq_mask[gt_rf_st:gt_rf_end])
                if np.logical_and(
                    trigger_RFmask_pb >= 0.5, 
                    trigger_RFmask_pb > trigger_EQmask_pb
                ):
                    pred_type = 'rockfall'

                elif np.logical_and(
                    len(trigger_EQmask) != 0,
                    trigger_EQmask_pb > trigger_RFmask_pb
                ):
                    eq_picks = picker(picker_args, pred_eq_mask, 
                        pred_eq_p, pred_eq_s)
                    if len(eq_picks) > 0:
                        pred_type = 'earthquake'
                    else:
                        pred_type = 'noise'
                else:
                    pred_type = 'noise'

            ### for earthquake labels
            elif label_type == 'earthquake':
                label_info = EQ_pick[k].numpy()
                try:
                    labeled_P_ntps = np.where(label_info.T[0]==1)[0][0]
                    labeled_S_ntps = np.where(label_info.T[1]==1)[0][0]
                    labeled_P = labeled_P_ntps*0.01
                    labeled_S = labeled_S_ntps*0.01
                    raw_tp_ts_diff = labeled_S-labeled_P
                    p_peak, p_value = pick_peaks(pred_eq_p, 
                            labeled_P, 0.01, search_win=1)
                    s_peak, s_value = pick_peaks(pred_eq_s, 
                            labeled_S, 0.01, search_win=1)
                except: 
                    # cannot find labeled P/S, 
                    # representing the pure noise seismogram
                    labeled_P = -999
                    labeled_S = -999
                    raw_tp_ts_diff = -999
                    p_peak, p_value = -999, -999
                    s_peak, s_value = -999, -999
                hp_p_snr, hp_s_snr = -999, -999

                # data type
                if np.logical_and(labeled_P==-999, labeled_S==-999):
                    lbl_type = 'noise'
                else:
                    lbl_type = 'earthquake'

                if lbl_type == 'noise':
                    # check eq mask
                    if np.logical_and(
                        len(trigger_EQmask) == 0, 
                        len(trigger_RFmask) == 0
                    ):
                        pred_type = 'noise'
                        trigger_EQmask_pb = 0.0
                        trigger_RFmask_pb = 0.0

                    elif len(trigger_EQmask) != 0:
                        eq_picks = picker(picker_args, pred_eq_mask, 
                            pred_eq_p, pred_eq_s)   

                        if len(eq_picks) > 0:
                            trigger_EQmask_pb = np.max([eq_picks[k][-1]
                                 for k in list(eq_picks.keys())]
                            )
                            pred_type = 'earthquake'
                        else:
                            pred_type = 'noise'

                    elif len(trigger_RFmask) != 0: 
                        trg_mean = np.array([np.mean(
                            pred_rf_mask[l[0]:l[1]]) for l in trigger_RFmask])
                        trg_mean = trg_mean[np.isnan(trg_mean)==False]
                        if len(trg_mean) == 0:
                            trigger_RFmask_pb = 0
                        else:
                            trigger_RFmask_pb = np.max(trg_mean)

                        if np.logical_and(
                            trigger_RFmask_pb >= 0.5, 
                            trigger_RFmask_pb > trigger_EQmask_pb
                        ):
                            pred_type = 'rockfall'

                elif lbl_type == 'earthquake':
                    trigger_EQmask_pb = np.round(np.mean(pred_eq_mask[
                        labeled_P_ntps:labeled_S_ntps]), 3)
                    trigger_RFmask_pb = np.round(np.mean(pred_rf_mask[
                        labeled_P_ntps:labeled_S_ntps]), 3)

                    if trigger_EQmask_pb >= 0.5:
                        pred_type = 'earthquake'
                    elif np.logical_and(
                            trigger_RFmask_pb >= 0.5, 
                            trigger_RFmask_pb > trigger_EQmask_pb
                        ):
                        pred_type = 'rockfall'
                    else:
                        pred_type = 'noise'
            # engineering, car label
            else: 
                if len(trigger_RFmask) != 0:
                    trg_mean = np.array(
                        [np.mean(pred_rf_mask[l[0]:l[1]]) 
                            for l in trigger_RFmask]
                    )
                    trg_mean = trg_mean[np.isnan(trg_mean)==False]
                    if len(trg_mean) == 0:
                        trigger_RFmask_pb = 0
                    else:
                        trigger_RFmask_pb = np.max(trg_mean)
                        pred_type = 'rockfall'

                elif len(trigger_EQmask) != 0:
                    eq_picks = picker(picker_args, pred_eq_mask, 
                        pred_eq_p, pred_eq_s)
                    
                    if len(eq_picks) > 0:
                        eq_keys = list(eq_picks.keys())
                        trigger_EQmask_pb = \
                            np.max([eq_picks[e][1] for e in eq_keys])
                        if trigger_EQmask_pb >= trigger_RFmask_pb:
                            pred_type = 'earthquake'
                        else:
                            pred_type = 'rockfall'
                    else:
                        pred_type = 'noise'
                else:
                    trigger_EQmask_pb = 0
                    trigger_RFmask_pb = 0
                    pred_type = 'noise'
            
            # write messages
            if T != 3:
                # rockfall, car, engineering messages
                message = f"{evid}, {label_type}, "+\
                        f"{pred_type}, {trigger_EQmask_pb:>5.2f}, "+\
                        f"{trigger_RFmask_pb:>5.2f}\n"
                f_out.write(message)
                ct += 1

            elif T == 3:
                dist = -999
                # earthquake message
                chn = 'EH'
                message = f"{evid}, {sta_info:>5s}, {chn:>3s}, "+\
                        f"{raw_tp_ts_diff:>11.2f}, "+\
                        f"{labeled_P:>5.2f}, {labeled_S:>5.2f}, "+\
                        f"{p_peak:>5.2f}, {p_value:>10.2f}, "+\
                        f"{s_peak:>5.2f}, {s_value:>10.2f}, "+\
                        f"{dist:>10.2f}, "+\
                        f"{hp_p_snr:8.2f}, {hp_s_snr:8.2f}, "+\
                        f"{lbl_type}, {pred_type}, "+\
                        f"{trigger_EQmask_pb:>5.2f}, "+\
                        f"{trigger_RFmask_pb:>5.2f}\n"
                f_out.write(message)
                ct += 1

            if plt_figure:
                # waveform
                gt_trc = trc[k].numpy().T
                trc_E, trc_N, trc_Z = gt_trc[0], gt_trc[1], gt_trc[2]
                x_t = np.arange(len(trc_Z))/100
                # compute spectrogram
                norm_spec = compute_STFT(trc_Z)
                spec_real = norm_spec[..., 0]
                spec_imag = norm_spec[..., 1]        
                t, sig_raw = ss.istft(
                    (spec_real.real + spec_imag.imag*1j), 
                    fs=100, nperseg=20, nfft=100, boundary='zeros')
                ff, tt, Sxx_raw_sig = ss.spectrogram(sig_raw, 
                        fs=100, nperseg=20, nfft=100)

                #---------- plot figure
                fig, ax = plt.subplots(9, 1, figsize=(10, 8))
                for j in range(8):
                    ax[j].set_xticks([])
                for l in range(4, 9):
                    ax[l].set_ylim(-0.1, 1.1)
                
                x_data = [trc_E, trc_N, trc_Z]
                for ll in range(3):
                    ax[ll].plot(t, x_data[ll], linewidth=1, color='k')
                ax[3].pcolormesh(tt, ff, np.abs(Sxx_raw_sig), 
                    shading='gouraud', cmap='hot', vmin=0, vmax=1)
                pred_data = [
                    pred_eq_p, pred_eq_s, pred_eq_mask, 
                    pred_rf_mask, gt_mask
                ]
                for jj in range(4, len(pred_data)+4):
                    ax[jj].plot(t, pred_data[jj-4], linewidth=1, color='b')
                pred_lbl = [
                    'E', 'N', 'Z', 'Freq.\n(Hz)',
                    'pred\nEQ_P', 'pred\nEQ_S', 'pred\nEQ_mask',
                    'pred\nRF_mask', 'ground-truth\nRF_mask'
                ]
                for kk in range(9):
                    ax[kk].set_ylabel(pred_lbl[kk])
                    ax[kk].set_xlim(0, 60)
                ax[-1].set_xlabel('Time (s)')

                plt.subplots_adjust(left=0.1,
                            bottom=0.08, 
                            right=0.95, 
                            top=0.95, 
                            wspace=0.1, 
                            hspace=0.1)
                ax[0].set_title(evid)
                plt.savefig(os.path.join(figout, evid+'.png'))
                plt.close()
                #plt.show()
    f_out.close()

