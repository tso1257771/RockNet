import os
import sys
sys.path.append('../../')
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from glob import glob
from copy import deepcopy
from obspy import read
from tools.data_utils import sac_len_complement
from tools.data_utils import gen_tar_func
from tools.data_utils import stream_standardize
from tools.RF_example_parser import write_TFRecord_RF_fusion
logging.basicConfig(level=logging.INFO,
        format='%(levelname)s : %(asctime)s : %(message)s')

basepath = '../'
wfdir = os.path.join(basepath, 'labeled_sac', 'EQ')
gen_type = 'val'
outpath = os.path.join(basepath, 
    'tfrecord_SingleSta', 'EQ', gen_type)
outfigpath = os.path.join(basepath, 
    'fig_tfrecord_SingleSta', 'EQ', gen_type)

os.system(f'rm -rf {outpath} {outfigpath}')
os.makedirs(outpath); os.makedirs(outfigpath)
# load labeling metadata
lbl_data = np.load(os.path.join(basepath, 'metadata',
    'eq_labels.npy'), allow_pickle=True)
lbl_idx = np.array(list(lbl_data.item()))
# load partition metadata
partition_data = np.load(os.path.join(basepath, 'metadata',
    'partition', 'eq_partition.npy'), allow_pickle=True)
eq_evid = partition_data.item()[gen_type]

center_sec = 60
data_length = 6000
### generate training data
train_ct = []
fig_ct = 0
sta_order = np.array(['LH01', 'LH02', 'LH03', 'LH04'])
for i in range(len(eq_evid)):
    evid = str(eq_evid[i])
    if os.path.exists(os.path.join(wfdir, evid)):
        logging.info(f"Generating {gen_type} data: {eq_evid[i]}")
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

    # make tfrecord
    

    net_data = []
    for m in range(len(sta_order)):
        if not (sta_order[m] in list(sta_meta.keys())):
            continue
        tfr_idx = f'{evid}_{sta_order[m]}_eq'

        wf_metadata = sta_meta[sta_order[m]]
        wf = deepcopy(wf_metadata['st'])
        eq_mask = deepcopy(wf_metadata['eq_mask'])
        p_tar = deepcopy(wf_metadata['eq_p'])
        s_tar = deepcopy(wf_metadata['eq_s'])
        wf_stt = wf[0].stats.starttime

        #  waveform data
        slice_stt = wf_stt+center_sec
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

        outfile = os.path.join(outpath, f'{tfr_idx}.tfrecord')
        write_TFRecord_RF_fusion(trc_3C, spectrogram,
            tar_EQpick, tar_EQmask, tar_RFmask, 
            idx=tfr_idx, outfile=outfile)
        
        if fig_ct < 300:
            fig, ax = plt.subplots(6, 1, figsize=(8, 6))
            xx = np.arange(data_length)*0.01
            trc_data = [seg_wf[0].data, seg_wf[1].data, seg_wf[2].data]
            eq_data = [EQpick_P, EQpick_S, seg_EQ_mask]
            ylbl = ['E', 'N', 'Z', 'Frequency', 'EQ', 'RF mask']
            for x in range(3):
                ax[x].plot(xx, trc_data[x])
                ax[4].plot(xx, eq_data[x])

            ax[3].pcolormesh(t, f, np.abs(norm_FT), cmap='viridis',
                shading='nearest')

            ax[5].plot(RFmask)
            for y in range(6):
                if y != 5:
                    ax[y].set_xticks([])
                ax[y].set_xlim(0, xx.max())
                ax[y].set_ylabel(ylbl[y])
            ax[4].set_ylim(-0.1, 1.1)
            ax[5].set_ylim(-0.1, 1.1)
            ax[5].set_xlabel('Time (s)')
            plt.tight_layout()
            plt.savefig(os.path.join(outfigpath, f'{tfr_idx}.png'))
            plt.close()
            fig_ct += 1
