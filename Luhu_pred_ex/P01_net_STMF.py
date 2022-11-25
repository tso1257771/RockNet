import os
import sys
sys.path.append('../')
sys.path.append('../tools/build_model')
import logging
import obspy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.signal as ss
import matplotlib.gridspec as gridspec
from glob import glob
from copy import deepcopy
from scipy.signal import tukey
from scipy.signal import find_peaks
from copy import deepcopy
from obspy import read, UTCDateTime
from tools.rockfall_net_STMF_fusion import compute_STFT
from tools.rockfall_net_STMF_fusion import RockFall_STMF
from tools.rockfall_net_STMF_fusion import sac_len_complement
from tools.build_model.rocknet_model import RockNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
logging.basicConfig(level=logging.INFO,
    format='%(levelname)s : %(asctime)s : %(message)s')

ddir = './sac/'
hdir = np.sort(np.unique(glob(os.path.join(ddir, '????.???.??'))))

odir = './net_pred'
mdl = '../trained_model/net_fusion_RF_60s/train.hdf5'
# prediction parameters
pred_interval_sec = 5
pred_npts = 6000
# load model
model = RockNet().associate_net(weights=mdl)
detector = RockFall_STMF(model=model, 
    pred_interval_sec=pred_interval_sec,
    pred_npts=pred_npts, dt=0.01, highpass=None)

wf_queue = dict()
max_npts = int(3600 + 120)*100
for i in range(len(hdir)):
    logging.info(f"Processing: {i+1}/{len(hdir)}: {hdir[i]}")
    dir_id = os.path.basename(hdir[i])
    hidx = dir_id.split('.')
    yr, jday, hr = int(hidx[0]), int(hidx[1]), int(hidx[2])
    stt = UTCDateTime(year=yr, julday=jday, hour=hr) - 60
    ent = stt + 60 + 3600 + 60

    # setup output parameters
    W_idx = f"Luhu.{dir_id}.sac"
    W_sta = ['RF', 'EQ', 'EH', 'EH', 'EH', 'EH']
    W_suffix = ['rfocc', 'eqocc', 'P', 'S', 'eqmask', 'rfmask']
    outdir = os.path.join(odir, dir_id)

    lbl = ['LH01', 'LH02', 'LH03', 'LH04']
    # iterate through all stations for the first time
    for w in range(4):
        try:
            st = read(os.path.join(ddir, dir_id, f'*{lbl[w]}.EHE.*'),
                starttime=stt, endtime=ent)
            st += read(os.path.join(ddir, dir_id, f'*{lbl[w]}.EHN.*'),
                starttime=stt, endtime=ent)
            st += read(os.path.join(ddir, dir_id, f'*{lbl[w]}.EHZ.*'),
                starttime=stt, endtime=ent)
            st = st.detrend('demean')
            st.filter('bandpass', freqmin=5, freqmax=45)

            stt = st[0].stats.starttime
            ent = st[0].stats.endtime
            st = sac_len_complement(
                #st.slice(stt+10*60, ent-10*60, nearest_sample=False)
                st, max_length=max_npts
            )
            wf_slices = np.stack([st[0].data, st[1].data, st[2].data], -1)
            wf_queue[lbl[w]] = wf_slices
        except:
            pass

    queued_sta_list = list(wf_queue.keys())
    if len(queued_sta_list) == 1:
        continue

    # complement data with random number if 
    for w in range(4):
        if not lbl[w] in queued_sta_list:
            wf_queue[lbl[w]] = np.random.random([max_npts, 3])

    # make waveform list
    wf_list = []
    for w in range(4):
        wf_list.append(wf_queue[lbl[w]])
    wf_list = np.array(wf_list)

    pred_rfocc, pred_eqocc, pred_P, pred_S, pred_eqmask, pred_rfmask =\
        detector.predict(wf_list, single_output=True)
    
    # output path and data
    W_data = [pred_rfocc, pred_eqocc, 
        pred_P, pred_S, pred_eqmask, pred_rfmask]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    

    chn = ['occ', 'occ', 'P', 'S', 'eqM', 'rfM']
    for W in range(len(W_data)):
        # write occurrence
        if W < 2:
            W_out = os.path.join(outdir, W_idx+f'.{W_suffix[W]}') 
            _wf = st[0].copy()
            _wf.stats.station = W_sta[W]
            _wf.stats.channel = chn[W]
            _wf.data = W_data[W]
            _wf.write(W_out, format='SAC')
        # write single-station outputs
        else:
            for ww in range(4):
                W_out = os.path.join(outdir, 
                    f"{lbl[ww]}.{dir_id}.sac"+f'.{W_suffix[W]}') 
                _wf = st[0].copy()
                _wf.stats.station = W_sta[W]
                _wf.stats.channel = chn[W]
                _wf.data = W_data[W][ww]
                _wf.write(W_out, format='SAC')                

