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
wfdir = os.path.join(basepath, 'labeled_sac', 'engineering')
gen_type = 'train'
outpath = os.path.join(basepath, 
    'tfrecord_SingleSta', 'engineering', gen_type)
outfigpath = os.path.join(basepath, 
    'fig_tfrecord_SingleSta', 'engineering', gen_type)

os.system(f'rm -rf {outpath} {outfigpath}')
os.makedirs(outpath); os.makedirs(outfigpath)

# load partition metadata
partition_data = np.load(os.path.join(basepath, 'metadata',
    'partition', 'engineering_partition.npy'), allow_pickle=True)
eng_evid = partition_data.item()[gen_type]

tar_evid = list()
for i in range(len(eng_evid)):
    evid = str(eng_evid[i])
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
        ### read waveform and make rockfall mask
        st = read(os.path.join(wfdir, evid, f'*.{stas[j]}.EH?.*.sac'))
        wf_stt = st[0].stats.starttime
        sta = stas[j]
        yr, jday, hr = wf_stt.year, wf_stt.julday, wf_stt.hour
        st.sort()
        info = st[0].stats
        
        slice_idx = [np.random.randint(info.npts - data_length) for _ in range(6)]
        for s in range(len(slice_idx)):
            #### make target functions first
            RF_mask = np.zeros(data_length)
            RF_unmask = np.ones(data_length) - RF_mask
            #tar_RFmask = np.array([RF_mask, RF_unmask]).T

            EQpick_P = np.zeros(data_length)
            EQpick_S = np.zeros(data_length)
            EQ_unpick = np.ones(data_length) - EQpick_P - EQpick_S
            #tar_EQpick = np.array([EQpick_P, EQpick_S, EQ_unpick]).T

            EQmask = np.zeros(data_length)
            EQ_unmask = np.ones(data_length) - EQmask
            #tar_EQmask = np.array([EQmask, EQ_unmask]).T
            
            center_pair_st = slice_idx[s]
            center_pair_ed = center_pair_st + data_length

            _st = deepcopy(st)
            _trc_E = _st[0].data[center_pair_st:center_pair_ed]
            _trc_N = _st[1].data[center_pair_st:center_pair_ed]
            _trc_Z = _st[2].data[center_pair_st:center_pair_ed]
            
            try:
                assert len(_trc_Z) == data_length
            except:
                continue

            for _s in [_trc_E, _trc_N, _trc_Z]:
                _s -= np.mean(_s)
                _s /= np.std(_s)
                _s[np.isnan(_s)] = 1e-8
                _s[np.isinf(_s)] = 1e-8

            trc_3C = np.array([_trc_E, _trc_N, _trc_Z]).T
            # STFT
            tmp_data = deepcopy(_trc_Z)
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

            tar_RFmask = np.array([RF_mask, RF_unmask]).T
            tar_EQpick = np.array([EQpick_P, EQpick_S, EQ_unpick]).T
            tar_EQmask = np.array( [EQmask, EQ_unmask]).T

            eng_idx = f'{evid}_{stas[j]}.{s+1:02}'
            outfile = os.path.join(outpath, eng_idx+'.tfrecord')

            write_TFRecord_RF_fusion(trc_3C, spectrogram,
                tar_EQpick, tar_EQmask, tar_RFmask, 
                idx=eng_idx, outfile=outfile)

            fig, ax = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
            x_data = [_trc_E, _trc_N, _trc_Z, tar_RFmask[:, 0]]
            ylbl = ['E', 'N', 'Z', 'RF mask']
            for x in range(4):
                ax[x].plot(x_data[x])
                ax[x].set_xlim(0, data_length)
                ax[x].set_ylabel(ylbl[x])
            ax[x].set_ylim(-0.1, 1.1)
            plt.tight_layout()
            plt.savefig(os.path.join(outfigpath, f'{eng_idx}.png'))
            #plt.show()
            plt.close()
            