import os
import sys
sys.path.append('../../')
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
from glob import glob
from copy import deepcopy
from obspy import read
from tools.data_utils import sac_len_complement
from tools.data_utils import gen_tar_func
from tools.data_utils import stream_standardize
from tools.network_RF_example_parser import write_TFRecord_RF_network
logging.basicConfig(level=logging.INFO,
        format='%(levelname)s : %(asctime)s : %(message)s')

gen_type = 'train'
basepath = '../'
wfdir = os.path.join(basepath, 'labeled_sac', 'EQ')
outpath = os.path.join(basepath, 
    'tfrecord_Association', 'EQ', gen_type)
outfigpath = os.path.join(basepath, 
    'fig_tfrecord_Association', 'EQ', gen_type)

os.system(f'rm -rf {outpath} {outfigpath}')
os.makedirs(outpath); os.makedirs(outfigpath)
# load labeling metadata
lbl_data = np.load(os.path.join(basepath, 'metadata',
    'eq_labels.npy'), allow_pickle=True)
lbl_idx = np.array(list(lbl_data.item()))
# load partition metadata
partition_data = np.load(os.path.join(basepath, 'metadata',
    'partition', 'eq_partition.npy'), allow_pickle=True)
rf_train_evid = partition_data.item()[gen_type]

center_sec = 60
data_length = 6000

### generate training data
train_ct = []
sta_order = np.array(['LH01', 'LH02', 'LH03', 'LH04'])
for i in range(len(rf_train_evid)):
    evid = str(rf_train_evid[i])
    if os.path.exists(os.path.join(wfdir, evid)):
        logging.info(f"Generating {gen_type} data: {rf_train_evid[i]}")
    else:
        continue

    stas = np.sort([os.path.basename(s).split('.')[1] for s in 
        glob(os.path.join(wfdir, evid, '*.EHZ.*.sac'))])
    unlbl_sta = sta_order[np.isin(sta_order, stas, invert=True)]

    sta_ct = 0
    sta_meta = dict()
    for j in range(len(stas)):
        sta_lbl = lbl_data.item()[f'{evid}_{stas[j]}']
        sta_lbl_center = np.mean(sta_lbl, axis=-1)

        # determine the centering label
        center_idx = np.array([np.logical_and(
            l[0]<=center_sec, center_sec<=l[1]) for l in sta_lbl])
        if  np.array_equal(center_idx, np.full(len(sta_lbl), False)):
            center_idx[
                np.argmin(np.abs(center_sec-sta_lbl_center))] = True

        ### read waveform and make earthquake mask
        st = read(os.path.join(wfdir, evid, f'*.{stas[j]}.EH?.*.sac'))
        trc_stt = st[0].stats.starttime
        eq_p_tar = gen_tar_func(st[0].stats.npts,
            np.round(sta_lbl[0][0]*100).astype(int), 20)
        eq_s_tar = gen_tar_func(st[0].stats.npts,
            np.round(sta_lbl[0][1]*100).astype(int), 30)
        eq_p_trc = deepcopy(st[0]); eq_p_trc.data = eq_p_tar
        eq_s_trc = deepcopy(st[0]); eq_s_trc.data = eq_s_tar

        EQ_mask = np.zeros(st[0].stats.npts)
        EQ_mask_trc = deepcopy(st[0])
        for l in range(len(sta_lbl)):
            center_lbl = sta_lbl[l]

            #### make target functions first
            rf_st_sec, rf_ent_sec = sta_lbl[l]
            rf_st = int(np.round(rf_st_sec, 2)*100)
            rf_ent = int(np.round(rf_ent_sec, 2)*100)
            _front = gen_tar_func(st[0].stats.npts, rf_st, 20)
            _front[np.argmax(_front)+1:] = 0

            if rf_ent != 12000:
                _back = gen_tar_func(st[0].stats.npts, rf_ent, 30)
                _back[:np.argmax(_back)-1] = 0

                mask = _front + _back
                mask[rf_st:rf_ent-1] = 1
            else:
                mask = np.zeros(st[0].stats.npts)
                mask[rf_st:] = 1
            EQ_mask += mask

        if EQ_mask.max() != 1:
            stop
            #continue
        if EQ_mask.min() != 0:
            stop
            #continue

        EQ_mask_trc.data = EQ_mask

        sta_meta[stas[j]] = dict()
        sta_meta[stas[j]]['st'] = st
        sta_meta[stas[j]]['eq_mask'] = EQ_mask_trc
        sta_meta[stas[j]]['eq_p'] = eq_p_trc
        sta_meta[stas[j]]['eq_s'] = eq_s_trc
        sta_meta[stas[j]]['center_label'] = center_lbl

    center_lbl_all = np.array([sta_meta[c]['center_label'] for c in stas])
    # make earthquake occurrence stream
    ev_stt = np.min(center_lbl_all[:, 0]) - 0.5
    last_lbl_ent = ev_stt + \
        (np.max(center_lbl_all[:, 1]) - np.min(center_lbl_all[:, 0]))
    eq_occ_st = deepcopy(st[0])
    eq_occ_st.data = gen_tar_func(st[0].stats.npts, int(ev_stt*100), 100)

    # fake up waveform of random floats for unlabeled stations
    for k in range(len(unlbl_sta)):
        _st = deepcopy(st)
        for _w in _st:
            _w.data = np.random.random(len(_w.data))
        _eq_mask_trc = deepcopy(st[0])
        _eq_mask_trc.data = np.zeros(len(st[0].data))
        _eq_p_trc = deepcopy(st[0])
        _eq_p_trc.data = np.zeros(len(st[0].data))
        _eq_s_trc = deepcopy(st[0])
        _eq_s_trc.data = np.zeros(len(st[0].data))
        sta_meta[unlbl_sta[k]] = dict()
        sta_meta[unlbl_sta[k]]['st'] = _st
        sta_meta[unlbl_sta[k]]['eq_mask'] = _eq_mask_trc
        sta_meta[unlbl_sta[k]]['eq_p'] = _eq_p_trc
        sta_meta[unlbl_sta[k]]['eq_s'] = _eq_s_trc
        sta_meta[unlbl_sta[k]]['center_label'] = [0, 0]
    
    # determine the center waveform with randomly distributed location
    available_sec = int(data_length*0.01) - (last_lbl_ent-ev_stt)
    center_start_sec = ev_stt - available_sec/2

    if np.logical_and(ev_stt > available_sec, available_sec>10) :
        avail_bef = available_sec
    else:
        avail_bef = ev_stt
        
    seg_front_start_sec = np.sort([ev_stt - 
        np.random.randint(int(avail_bef*100))*0.01 
            for C in range(2)])


    seg_back_stt_sec = np.sort(np.array([ 
            np.random.uniform(
                low=center_start_sec,
                high=ev_stt
            ) for C in range(2)]))
    assert np.all(seg_back_stt_sec > center_start_sec)

    # remove trace with insufficient data 
    if gen_type == 'test':
        seg_start_sec = np.array([center_start_sec])
    else:
        seg_start_sec = np.hstack([
            center_start_sec, seg_back_stt_sec, seg_front_start_sec])

    seg_start_sec = seg_start_sec[
        np.logical_and(
            seg_start_sec>0,
            (seg_start_sec+int(data_length*0.01)) < len(st[0].data)*0.01)
        ]
    seg_start_sec = np.sort(seg_start_sec)

    # make tfrecord
    for s in range(len(seg_start_sec)):
        ev_occ_npts = int(100*(ev_stt - seg_start_sec[s]))
        eq_occ_pdf = gen_tar_func(data_length, ev_occ_npts, 100)
        rf_occ_pdf = np.zeros(data_length)

        tar_eq_occ = np.array([
            eq_occ_pdf, np.ones(data_length)-eq_occ_pdf]).T
        tar_rf_occ = np.array([
            rf_occ_pdf , np.ones(data_length)-rf_occ_pdf ]).T

        tfr_idx = f'{evid}_eq_{s+1:02}'

        net_data = []
        for m in range(len(sta_order)):
            wf_metadata = sta_meta[sta_order[m]]
            wf = deepcopy(wf_metadata['st'])
            eq_mask = deepcopy(wf_metadata['eq_mask'])
            p_tar = deepcopy(wf_metadata['eq_p'])
            s_tar = deepcopy(wf_metadata['eq_s'])
            wf_stt = wf[0].stats.starttime

            #  waveform data
            slice_stt = wf_stt+seg_start_sec[s]
            slice_ent = slice_stt + data_length*0.01

            seg_wf = stream_standardize(
                sac_len_complement(
                    deepcopy(wf).slice(
                        starttime=slice_stt, endtime=slice_ent),
                    max_length=data_length),
                data_length=data_length
            )

            #seg_wf.plot()
            seg_p_tar_trc = p_tar.slice(slice_stt, slice_ent,
                nearest_sample=False)
            seg_s_tar_trc = s_tar.slice(slice_stt, slice_ent,
                nearest_sample=False)
            seg_EQ_mask_trc = eq_mask.slice(slice_stt, slice_ent,
                nearest_sample=False)   

            trc_3C = np.array([
                seg_wf[0].data, seg_wf[1].data, seg_wf[2].data]).T

            # STFT
            tmp_data = deepcopy(seg_wf[2].data)
            tmp_mean = np.mean(tmp_data)
            tmp_data -= tmp_mean

            sos = ss.butter(4, 0.1, 'high', fs=100, output='sos')
            tmp_data = ss.sosfilt(sos, tmp_data)

            f, t, tmp_FT = ss.stft(tmp_data, fs=100, nperseg=20, 
                nfft=100, boundary='zeros')
            tmp_std = np.std(tmp_FT)
            norm_FT = tmp_FT/tmp_std
            norm_FT_real, norm_FT_imag = norm_FT.real, norm_FT.imag
            norm_FT_real[np.isnan(norm_FT_real)]=0
            norm_FT_imag[np.isnan(norm_FT_imag)]=0
            norm_FT_real[np.isinf(norm_FT_real)]=0
            norm_FT_imag[np.isinf(norm_FT_imag)]=0
            spectrogram = np.stack([norm_FT_real, norm_FT_imag], -1)

            # earthquake 
            seg_EQ_mask = seg_EQ_mask_trc.data[:data_length]
            seg_EQ_unmask = np.ones(data_length) - seg_EQ_mask
            tar_EQmask = np.array([seg_EQ_mask, seg_EQ_unmask]).T

            # EQpick
            EQpick_P = seg_p_tar_trc.data[:data_length]
            EQpick_S = seg_s_tar_trc.data[:data_length]
            EQ_unpick = np.ones(data_length) - EQpick_P - EQpick_S
            tar_EQpick = np.array([EQpick_P, EQpick_S, EQ_unpick]).T

            # RFmask
            RFmask = np.zeros(data_length)
            RF_unmask = np.ones(data_length) - RFmask
            tar_RFmask = np.array([RFmask, RF_unmask]).T

            net_data.append([
                trc_3C, spectrogram, tar_EQpick, tar_EQmask, tar_RFmask])
        
        # write tfrecord of general order data
        ordered_net_data = np.array(net_data, dtype=object)
        stack_net_data = np.hstack(ordered_net_data)
        write_TFRecord_RF_network(stack_net_data, tar_rf_occ, tar_eq_occ,
            outfile=os.path.join(outpath, f"{tfr_idx}.tfrecord"))
        # write tfrecord of random order data
        if gen_type != 'test':
            r_net_order = np.random.permutation(np.arange(4))
            random_net_data = ordered_net_data[r_net_order]
            r_stack_net_data = np.hstack(random_net_data)
            write_TFRecord_RF_network(r_stack_net_data, tar_rf_occ, tar_eq_occ,
                outfile=os.path.join(outpath, f"{tfr_idx}_random.tfrecord"))

        # plot figures
        sta_trc_Z = np.array([Z.T[2] for Z in stack_net_data[::5]])
        sta_eq_mask = np.array([Z.T[0] for Z in stack_net_data[3::5]])
        sta_p_tar = np.array([Z.T[0] for Z in stack_net_data[2::5]])
        sta_s_tar = np.array([Z.T[1] for Z in stack_net_data[2::5]])
        eq_occ = tar_eq_occ.T[0] 
        x = np.arange(len(eq_occ))/100
        fig, ax = plt.subplots(4, 1, figsize=(10, 8))
        for a in range(4):
            ax[a].plot(x, sta_trc_Z[a], linewidth=1, color='k')
            ax_t = ax[a].twinx()
            ax_t.plot(x, sta_eq_mask[a], linewidth=1, color='b')
            ax_t.plot(x, sta_p_tar[a], linewidth=1, color='r')
            ax_t.plot(x, sta_s_tar[a], linewidth=1, color='g')
            ax_t.set_ylim(-0.1, 1.1)
            ax[a].set_ylabel(sta_order[a])
            if a != 3:
                ax[a].set_xticks([])
            ax[a].set_xlim(0, data_length*0.01)
        
        ax[3].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(outfigpath, f'{tfr_idx}.png'))
        plt.close()
        #plt.show()