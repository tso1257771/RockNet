import numpy as np
import scipy.signal as ss
from copy import deepcopy
from scipy.signal import tukey
from scipy.signal import find_peaks
from obspy import read
from obspy.signal import filter
from obspy.signal.filter import envelope
from obspy.signal.trigger import trigger_onset

def compute_envelope(wf, dt=0.01, mva_win_sec=3):
    # take obspy 3C stream as input, 
    # establish moving average function of filtered envelope function
    wf_env = deepcopy(wf).detrend('demean')
    arr_shape = len(wf_env), len(wf_env[0].data)
    mva_win = int(mva_win_sec/dt)
    st_env = np.zeros(arr_shape)
    env_mva = np.zeros(arr_shape)
    for ct, p in enumerate(wf_env):
        st_env[ct] = envelope(p.data)
        mva = np.convolve(st_env[0],
            np.ones((mva_win,))/mva_win, mode='valid')
        env_mva[ct][-len(mva):] = mva
        wf_env[ct].data[:len(wf_env[0].data)-mva_win] =\
             env_mva[ct][mva_win:]
        wf_env[ct].data[-mva_win:] = np.zeros(mva_win)
    return wf_env