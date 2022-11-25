import os
import sys
sys.path.append('../')
sys.path.append('../tools/build_model')
import shutil
import logging
import numpy as np
import scipy.signal as ss
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import find_peaks
from obspy.signal.trigger import trigger_onset
from tools.rockfall_STMF_fusion import compute_STFT
from tools.build_model.rocknet_model import RockNet
from tools.network_RF_example_parser import tfrecord_dataset_fusion_RF_net
from tools.EQ_picker import picker
from tools.data_utils import pick_peaks
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
logging.basicConfig(level=logging.INFO,
    format='%(levelname)s : %(asctime)s : %(message)s')

basepath = '../'
datapath = '../data/tfrecord_Association'
## CHANGE the variable mdl_hdr for different models, i.e.,
# 'net_fusion_RF_60s' for the association model trained with INSTANCE,
# 'net_fusion_RF_60s_noINSTANCE' trained without INSTANCE
mdl_hdr = 'net_fusion_RF_60s_noINSTANCE'
RF_model = f'../trained_model/{mdl_hdr}/train.hdf5'
model = RockNet().associate_net(weights=RF_model)
outpath = os.path.join('./pred_results', mdl_hdr)
if os.path.exists(outpath):
    shutil.rmtree(outpath)
os.makedirs(outpath)

plt_figure = False
message_hdr = 'evid, label_type, pred_type, ev_occ_pt, pred_occ_pt, '+\
    'eqocc_prob, rfocc_prob\n'

tfr_dir = [
    os.path.join(datapath, 'RF/test'),
    os.path.join(datapath, 'car/test'),
    os.path.join(datapath, 'engineering/test'),
    os.path.join(datapath, 'EQ/test')
]
tfr_lbl = [
    'rockfall', 'nz_car', 'nz_engineering', 'earthquake'
]

occ_detect_thre = 0.5
for T in range(len(tfr_dir)):
    label_type = tfr_lbl[T]
    tfr_glob = np.sort(glob(os.path.join(tfr_dir[T], '*.tfrecord')))

    message_out = os.path.join(outpath, f'Association_pred_{label_type}.txt')
    figout = os.path.join(outpath, f'fig_Association_pred_{label_type}')
    if plt_figure:
        os.makedirs(figout)
    f_out = open(message_out, 'w')
    f_out.write(message_hdr)

    fig_ct = 0
    split_iter = np.ceil(len(tfr_glob)/300).astype(int)
    sep_idx = int(len(tfr_glob)//split_iter)
    for i in range(split_iter):
        logging.info(f"Iterating {label_type}: {i+1}/{split_iter}")

        st_idx, end_idx = sep_idx*i, sep_idx*(i+1)
        if i != split_iter-1:
            pred_batch_list = tfr_glob[st_idx:end_idx]
            n_pred = len(pred_batch_list)

        else:
            pred_batch_list = tfr_glob[st_idx:]
            n_pred = len(pred_batch_list)
        pred_batch = iter(tfrecord_dataset_fusion_RF_net(
            pred_batch_list,
            repeat=1, data_length=6000, 
            spec_signal_shape=(51, 601, 2),
            batch_size=n_pred, shuffle=False))

        trc, spec, gt_pick, gt_eqmask, gt_rfmask, \
            gt_eqocc, gt_rfocc = next(pred_batch)
        # make predictions
        pred_rfocc, pred_eqocc, pred_eqpick, \
            pred_eqmask, pred_rfmask = model.predict([trc, spec])
        
        for k in range(n_pred):
            logging.info(f"Processing {k+1}/{n_pred}")
            idx = '.'.join(
                os.path.basename(pred_batch_list[k]).split('_')[:2])

            sta_trc = trc[k].numpy()
            sta_spec = spec[k].numpy()
            sta_gt_pick = gt_pick[k].numpy()
            sta_gt_eqmask = gt_eqmask[k].numpy()
            sta_gt_rfmask = gt_rfmask[k].numpy()
            lbl_eqocc = gt_eqocc[k].numpy().T[0]
            lbl_rfocc = gt_rfocc[k].numpy().T[0]
            rfocc = pred_rfocc[k].T[0]
            eqocc = pred_eqocc[k].T[0]

            if label_type in ['rockfall', 'earthquake']:
                if label_type == 'rockfall':
                    pt_occ = np.argmax(lbl_rfocc)*0.01
                elif label_type == 'earthquake':
                    pt_occ = np.argmax(lbl_eqocc)*0.01
                # search range og ground truth
                search_occ_idx = np.array([pt_occ-20, pt_occ+20])
                if search_occ_idx[0] <= 0:
                    search_occ_idx[0] = 0

                # find max value of rfocc around ground truth
                search_idx_s = int(search_occ_idx[0]*100)
                search_idx_e = int(search_occ_idx[1]*100)
                pred_rfocc_argmax = np.argmax(
                    rfocc[search_idx_s:search_idx_e]
                )
                pred_rfocc_value = rfocc[search_idx_s+pred_rfocc_argmax]
                # find max value of eqocc around ground truth
                pred_eqocc_argmax = np.argmax(
                    eqocc[search_idx_s:search_idx_e]
                )
                pred_eqocc_value = eqocc[search_idx_s+pred_eqocc_argmax]

                gt_occ_pt = int(pt_occ*100)

                if np.logical_and(
                    pred_rfocc_value > pred_eqocc_value, 
                    pred_rfocc_value >= occ_detect_thre
                ):
                    pred_type = 'rockfall'
                    pred_occ_pt = search_idx_s + pred_rfocc_argmax

                elif np.logical_and(
                    pred_eqocc_value > pred_rfocc_value, 
                    pred_eqocc_value >= occ_detect_thre
                ):
                    pred_type = 'earthquake'
                    
                    pred_occ_pt = search_idx_s + pred_eqocc_argmax
                else:
                    pred_type = 'noise'
                    pred_occ_pt = -1
            # noise label
            elif label_type in ['nz_engineering', 'nz_car']:
                # find max value of rfocc 
                pred_rfocc_argmax = np.argmax(rfocc)
                pred_rfocc_value = rfocc[pred_rfocc_argmax]
                # find max value of eqocc 
                pred_eqocc_argmax = np.argmax(eqocc)
                pred_eqocc_value = eqocc[pred_eqocc_argmax]

                if np.logical_and(
                    pred_rfocc_value > pred_eqocc_value, 
                    pred_rfocc_value >= occ_detect_thre
                ):
                    pred_type = 'rockfall'
                    pred_occ_pt = pred_rfocc_argmax

                elif np.logical_and(
                    pred_eqocc_value > pred_rfocc_value, 
                    pred_eqocc_value >= occ_detect_thre
                ):
                    pred_type = 'earthquake'
                    
                    pred_occ_pt = pred_eqocc_argmax
                else:
                    pred_type = 'noise'
                    pred_occ_pt = -1
            message = f"{idx}, {label_type}, {pred_type}, "+\
                    f"{gt_occ_pt}, {pred_occ_pt}, "+\
                    f"{pred_eqocc_value:.2f}, {pred_rfocc_value:.2f}\n"
            f_out.write(message)

            if np.logical_and(plt_figure, fig_ct < 100):
                # plot figures 
                x = np.arange(6000)*0.01
                fig, ax = plt.subplots(6, 1, figsize=(10, 8))
                trc_Z =  [sta_trc[p].T[2] for p in range(4)]
                sta_p = [pred_eqpick[k][p].T[0] for p in range(4)]
                sta_s = [pred_eqpick[k][p].T[1] for p in range(4)]
                sta_eqmask = [pred_eqmask[k][p].T[0] for p in range(4)]
                sta_rfmask = [pred_rfmask[k][p].T[0] for p in range(4)]
                #eqocc = pred_eqocc[0].T[0]

                ylbl = ['LH01', 'LH02', 'LH03', 'LH04','RFocc', 'EQocc']
                for j in range(4):
                    ax[j].plot(x, trc_Z[j], 
                        linewidth=1, color='gray')
                    ax_t = ax[j].twinx()
                    ax_t.plot(x, sta_p[j], 
                        linewidth=1, color='b', label='P')
                    ax_t.plot(x, sta_s[j], 
                        linewidth=1, color='r',label='S')
                    ax_t.plot(x, sta_eqmask[j], 
                        linewidth=1, color='g',label='EQmask')
                    ax_t.plot(x, sta_rfmask[j], 
                        linewidth=1, color='brown',label='rfmask')
                    ax_t.set_ylim(-0.1, 1.1)
                ax[4].plot(x, rfocc, linewidth=1)
                ax[4].set_ylim(-0.1, 1.1)

                ax[5].plot(x, eqocc, linewidth=1)
                ax[5].set_ylim(-0.1, 1.1)
                
                if label_type == 'earthquake':
                    ax[5].plot(x, lbl_eqocc, 
                        linewidth=1, linestyle=':', color='r')
                    ax[5].axvspan(search_occ_idx[0], search_occ_idx[1], 
                        color='pink', alpha=0.3)
                elif label_type == 'rockfall':
                    ax[4].plot(x, lbl_rfocc, 
                        linewidth=1, linestyle=':', color='r')
                    ax[4].axvspan(search_occ_idx[0], search_occ_idx[1], 
                        color='pink', alpha=0.3)

                ax[5].set_xlabel("Time (s)")
                for k in range(6):
                    ax[k].set_xlim(0, x.max())
                    ax[k].set_ylabel(ylbl[k])
                    if k != 5:
                        ax[k].set_xticks([])

                plt.tight_layout()
                #plt.show()

                outid = f'{idx}.png'
                out_name = os.path.join(figout, outid)
                plt.savefig(out_name)
                plt.close()
                fig_ct += 1                
    f_out.close()
