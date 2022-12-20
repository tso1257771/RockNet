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
wfdir = os.path.join(basepath, 'labeled_sac', 
    'car_multiple_sta_label')
gen_type = 'test'
outpath = os.path.join(basepath, 
    'tfrecord_SingleSta', 'car_multiple_sta_label', gen_type)
outfigpath = os.path.join(basepath, 
    'fig_tfrecord_SingleSta', 'car_multiple_sta_label', gen_type)
os.system(f'rm -rf {outpath} {outfigpath}')
os.makedirs(outpath); os.makedirs(outfigpath)
# load labeling metadata
lbl_data = np.load(os.path.join(basepath, 'metadata',
    'car_labels.npy'), allow_pickle=True)
lbl_idx = np.array(list(lbl_data.item()))
# load partition metadata
partition_data = np.load(os.path.join(basepath, 'metadata',
    'partition', 'car_partition.npy'), allow_pickle=True)
car_evid = partition_data.item()[gen_type]

tar_evid = list()
for i in range(len(car_evid)):
    evid = str(car_evid[i])
    if os.path.exists(os.path.join(wfdir, evid)):
        tar_evid.append(evid)
    else:
        continue
tar_evid = np.array(tar_evid)

center_sec = 60
data_length = 6000
### generate training data
total_ct = []
for i in range(len(tar_evid)):
    evid = str(tar_evid[i])
    if os.path.exists(os.path.join(wfdir, evid)):
        logging.info(f"Generating {gen_type} data: "+\
            f"{i+1}/{len(tar_evid)}: {tar_evid[i]}")
    else:
        continue

    stas = np.sort([os.path.basename(s).split('.')[1] for s in 
        glob(os.path.join(wfdir, evid, '*.EHZ.*.sac'))])

    sta_ct = 0
    for j in range(len(stas)):
        sta_lbl = lbl_data.item()[f'{evid}_{stas[j]}']
        sta_lbl_center = np.mean(sta_lbl, axis=-1)

        # determine the centering label
        center_idx = np.array([np.logical_and(
            l[0]<=center_sec, center_sec<=l[1]) for l in sta_lbl])
        if  np.array_equal(center_idx, np.full(len(sta_lbl), False)):
            center_idx[
                np.argmin(np.abs(center_sec-sta_lbl_center))] = True

        ### read waveform and make rockfall mask
        st = read(os.path.join(wfdir, evid, f'*.{stas[j]}.EH?.*.sac'))
        trc_stt = st[0].stats.starttime
        car_mask = np.zeros(st[0].stats.npts)
        car_mask_trc = deepcopy(st[0])
        chosed_lbl = []
        for l in range(len(sta_lbl)):
            # choose potential complete car waveform
            if np.logical_and(
                sta_lbl[l][1] - sta_lbl[l][0] > 5, sta_lbl[l][0] > 10):
                chosed_lbl.append(sta_lbl[l])

            #### make target functions first
            car_st_sec, car_ent_sec = sta_lbl[l]
            car_st = int(np.round(car_st_sec, 2)*100)
            car_ent = int(np.round(car_ent_sec, 2)*100)
            _front = gen_tar_func(st[0].stats.npts, car_st, 100)
            _front[np.argmax(_front)+1:] = 0

            _back = gen_tar_func(st[0].stats.npts, car_ent, 100)
            _back[:np.argmax(_back)-1] = 0

            mask = _front + _back
            mask[car_st:car_ent-1] = 1

            car_mask += mask

        if car_mask.max() > 1:
            continue

        car_mask_trc.data = car_mask
        chosed_lbl = np.array(chosed_lbl)

        collect_start_sec = []
        ### centering the chosed labeling waveform      
        for c in range(len(chosed_lbl)):
            base_sec = chosed_lbl[c][1] - chosed_lbl[c][0]
            max_bef_sec = np.floor(chosed_lbl[c][0])
            max_aft_sec = len(st[0].data)*0.01 - np.ceil(chosed_lbl[c][1])
            available_sec = np.floor(data_length*0.01 - base_sec)
            if available_sec > 1:
                if np.logical_and(max_bef_sec > available_sec//2,
                        max_aft_sec > available_sec//2):
                    center_start_sec = chosed_lbl[c][0] - available_sec//2
                elif np.logical_and(max_bef_sec > available_sec//2,
                        max_aft_sec < available_sec//2):
                    center_start_sec = len(st[0].data)*0.01 - data_length*0.01
                elif np.logical_and(max_bef_sec < available_sec//2,
                        max_aft_sec > available_sec//2):
                    center_start_sec = 0
            else:
                center_start_sec = np.mean(chosed_lbl[c]) - data_length*0.01//2
            collect_start_sec.append(center_start_sec)

        start_utc = trc_stt + collect_start_sec[0]
        end_utc = start_utc + data_length*0.01
        new_evid = f'{start_utc.year:04}{start_utc.julday:03}'+\
            f'{start_utc.hour:02}{start_utc.minute:02}{start_utc.second:02}'


        seg_wf = stream_standardize(
                sac_len_complement(
                    deepcopy(st).slice(
                        starttime=start_utc, 
                        endtime=end_utc),
                max_length=data_length),
            data_length=data_length)

        seg_car_mask_trc = deepcopy(car_mask_trc).slice(
                    starttime=start_utc, 
                    endtime=end_utc,
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
            
        # rockfall 
        #seg_RF_mask = seg_RF_mask_trc.data[:data_length]
        #seg_RF_unmask = np.ones(data_length) - seg_RF_mask
        seg_RF_mask = np.zeros(data_length)
        seg_RF_unmask = np.ones(data_length) - seg_RF_mask
        tar_RFmask = np.array([seg_RF_mask, seg_RF_unmask]).T

        # EQpick
        EQpick_P = np.zeros(data_length)
        EQpick_S = np.zeros(data_length)
        EQ_unpick = np.ones(data_length) - EQpick_P - EQpick_S
        tar_EQpick = np.array([EQpick_P, EQpick_S, EQ_unpick]).T

        # EQmask
        EQmask = np.zeros(data_length)
        EQ_unmask = np.ones(data_length) - EQmask
        tar_EQmask = np.array([EQmask, EQ_unmask]).T

        # output files
        RF_idx = f'{new_evid}_{stas[j]}'
        outfile = os.path.join(outpath, RF_idx+'.tfrecord')


        write_TFRecord_RF_fusion(trc_3C, spectrogram,
            tar_EQpick, tar_EQmask, tar_RFmask, 
            idx=RF_idx, outfile=outfile)

        sta_ct += 1
        
        fig, ax = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
        x_data = [seg_wf[0].data, seg_wf[1].data, 
            seg_wf[2].data, tar_RFmask[:, 0]]
        ylbl = ['E', 'N', 'Z', 'RF mask']
        for x in range(4):
            ax[x].plot(x_data[x])
            ax[x].set_xlim(0, data_length)
            ax[x].set_ylabel(ylbl[x])
        ax[x].set_ylim(-0.1, 1.1)
        ax[x].set_xlabel('npts')
        ax[0].set_title(RF_idx)
        plt.tight_layout()
        plt.savefig(os.path.join(outfigpath, f'{RF_idx}.png'))
        #plt.show()
        plt.close()
    total_ct.append(sta_ct)


