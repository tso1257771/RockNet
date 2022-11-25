import numpy as np
import scipy.signal as ss
from obspy import read
from obspy.signal import filter
from scipy.signal import tukey
from obspy.signal.trigger import trigger_onset
from scipy.signal import find_peaks
from copy import deepcopy

def snr_pt_v2(tr_vertical, tr_horizontal, pt_p, pt_s, mode='std',
            snr_pre_window=5, snr_post_window=5, highpass=None):
    """
    Calculate snr
    tr_vertical: sac trace vertical component
    tr_horizontal: sac trace horizontal component
    pt_p: p phase utcdatetime object
    pt_s: s phase udtdatetime object
    """
    if highpass:
        tr_vertical = tr_vertical.filter(
            'highpass', freq=highpass).\
            taper(max_percentage=0.1, max_length=0.1)
        tr_horizontal = tr_horizontal.filter(
            'highpass', freq=highpass).\
            taper(max_percentage=0.1, max_length=0.1)
    tr_signal_p = tr_vertical.copy().slice( 
        pt_p, pt_p + snr_pre_window )
    tr_signal_s = tr_horizontal.copy().slice( 
        pt_s, pt_s + snr_pre_window ) 
    tr_noise_p = tr_vertical.copy().slice( 
        pt_p - snr_pre_window, pt_p )
    tr_noise_s = tr_horizontal.copy().slice( 
        pt_s-snr_pre_window, pt_s )
  
    if mode.lower() == 'std':
        snr_p = np.std(tr_signal_p.data)/np.std(tr_noise_p.data)
        snr_s = np.std(tr_signal_s.data)/np.std(tr_noise_s.data)

    elif mode.lower() == 'sqrt':
        snr_p = np.sqrt(np.square(tr_signal_p.data).sum())\
            / np.sqrt(np.square(tr_noise_p.data).sum()) 
        snr_s = np.sqrt(np.square(tr_signal_s.data).sum())\
            / np.sqrt(np.square(tr_noise_s.data).sum()) 

    return snr_p, snr_s

def compute_STFT(data):
    # STFT
    tmp_data = deepcopy(data)
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
    return spectrogram

def conti_standard_wf_fast(wf, pred_npts, pred_interval_sec, dt, 
        pad_zeros=True):
    '''
    input: 
    wf: obspy.stream object (raw_data)
    pred_npts
    pred_interval_sec
    pad_zeros: pad zeros before after the waveform for full repeating predictions 

    output:
    wf_slices
    wf_start_utc
    '''
    assert len(np.unique([len(w) for w in wf])) == 1

    raw_n = len(wf[0].data)
    pred_rate = int(pred_interval_sec/dt)
    pad_bef = pred_npts - pred_rate
    pad_aft = pred_npts
    pad_wf = []
    for W in wf:
        pad_wf.append(
            np.hstack([
            np.repeat(W[0], pad_bef),
            W,
            np.repeat(W[-1], pad_aft)
        ])
    )
    pad_wf = np.array(pad_wf)

    full_len = len(pad_wf[0])
    n_marching_win = int((full_len - pred_npts)/pred_rate)+1
    n_padded = full_len - raw_n


    wf_n = []
    for w in range(3):
        wf_ = np.array([deepcopy(pad_wf[w][pred_rate*i:pred_rate*i+pred_npts]) 
            for i in range(n_marching_win)])

        wf_dm = np.array([i-np.mean(i) for i in wf_])
        wf_std = np.array([np.std(i) for i in wf_dm])
        # reset std of 0 to 1 to prevent from ZeroDivisionError
        wf_std[wf_std==0] = 1
        wf_norm = np.array([wf_dm[i]/wf_std[i] for i in range(len(wf_dm))])
        wf_n.append(wf_norm)

    spectrogram_ = np.array(
        [compute_STFT(w_norm) for w_norm in wf_norm]
    )

    wf_slices = np.stack([wf_n[0], wf_n[1], wf_n[2]], -1)


    return wf_slices, spectrogram_, pad_bef, pad_aft

def sac_len_complement(wf, max_length=None):
    '''Complement sac data into the same length
    '''
    wf_n = np.array([len(i.data) for i in wf])
    if not max_length:
        max_n = np.max(wf_n)
    else:
        max_n = max_length

    append_wf_id = np.where(wf_n!=max_n)[0]
    for w in append_wf_id:
        append_npts = max_n - len(wf[w].data)
        if append_npts > 0:
            for p in range(append_npts):
                wf[w].data = np.insert(
                    wf[w].data, len(wf[w].data), np.zeros(len(wf[w].data)))
        elif append_npts < 0:
            wf[w].data = wf[w].data[:max_n]
    return wf
          
def stream_standardize(st, data_length):
    '''
    input: obspy.stream object (raw data)
    output: obspy.stream object (standardized)
    '''
    data_len = [len(i.data) for i in st]
    check_len = np.array_equal(data_len, np.repeat(data_length, 3))
    if not check_len:
        for s in st:
            res_len = len(s.data) - data_length
            if res_len > 0:
                s.data = s.data[:data_length]
            elif res_len < 0:
                last_pt = 0#s.data[-1]
                s.data = np.insert(s.data, -1, np.repeat(last_pt, -res_len))

    st = st.detrend('demean')
    for s in st:
        data_std = np.std(s.data)
        if data_std == 0:
            data_std = 1
        s.data /= data_std
        
        if np.isinf(s.data).any():
            s.data[np.isinf(s.data)] = 1e-4
        if np.isnan(s.data).any():
            s.data[np.isnan(s.data)] = 1e-4 
    return st

def wf_seg_standardize(wf, st_idx, end_idx, pred_npts=2001):
    #Z-score standardization
    
    # standardize waveform segment for model prediction
    wf_E = wf[0].data[st_idx:end_idx]
    wf_N = wf[1].data[st_idx:end_idx]
    wf_Z = wf[2].data[st_idx:end_idx]
    wf_E -= np.mean(wf_E); wf_E/=np.std(wf_E)
    wf_N -= np.mean(wf_N); wf_N/=np.std(wf_N)
    wf_Z -= np.mean(wf_Z); wf_Z/=np.std(wf_Z)  

    wfs_1 = np.array([wf_E, wf_N, wf_Z]).T
    assert wfs_1.shape == (pred_npts, 3)
    return wfs_1

def pred_MedianFilter(eqpick, eqmask, rfmask, wf_npts, dt, 
        pred_npts, pred_interval_sec, pad_bef, pad_aft):
    ### 3. Integrate continuous predictions
    wf_n = wf_npts + (pad_bef+pad_aft)
    pred_array_P = [[] for _ in range(wf_n)]
    pred_array_S = [[] for _ in range(wf_n)]
    pred_array_eqmask = [[] for _ in range(wf_n)]
    pred_array_rfmask = [[] for _ in range(wf_n)]

    pred_interval_pt =  int(round(pred_interval_sec/dt))

    init_pt = 0
    for i in range(len(eqpick)):
        pp = np.array_split(eqpick[i].T[0], pred_npts)
        ss = np.array_split(eqpick[i].T[1], pred_npts)
        eqmm = np.array_split(eqmask[i].T[0], pred_npts)
        rfmm = np.array_split(rfmask[i].T[0], pred_npts)

        j = 0
        for p, s, eqm, rfm in zip(pp, ss, eqmm, rfmm):
            pred_array_P[init_pt+j].append(p)
            pred_array_S[init_pt+j].append(s)
            pred_array_eqmask[init_pt+j].append(eqm)
            pred_array_rfmask[init_pt+j].append(rfm)
            j += 1
        init_pt += pred_interval_pt
    
    pred_array_P = np.array(pred_array_P, dtype='object')
    pred_array_S = np.array(pred_array_S, dtype='object')
    pred_array_eqmask = np.array(pred_array_eqmask, dtype='object')
    pred_array_rfmask = np.array(pred_array_rfmask, dtype='object')
    # fast revision of bottleneck
    lenM = np.array([len(m) for m in pred_array_rfmask])
    nums = np.unique(lenM)
    array_P_med = np.zeros(wf_n)
    array_S_med = np.zeros(wf_n)
    array_eqM_med = np.zeros(wf_n)
    array_rfM_med = np.zeros(wf_n)
    for k in nums:
        num_idx = np.where(lenM==k)[0]
        array_P_med[num_idx] = \
            np.median(np.hstack(
                np.take(pred_array_P, num_idx)), axis=0)
        array_S_med[num_idx] = \
            np.median(np.hstack(
                np.take(pred_array_S, num_idx)), axis=0)
        array_eqM_med[num_idx] = \
            np.median(np.hstack(
                np.take(pred_array_eqmask, num_idx)), axis=0)
        array_rfM_med[num_idx] = \
            np.median(np.hstack(
                np.take(pred_array_rfmask, num_idx)), axis=0)

    array_P_med = array_P_med[pad_bef:-pad_aft]
    array_S_med = array_S_med[pad_bef:-pad_aft]
    array_eqM_med = array_eqM_med[pad_bef:-pad_aft]
    array_rfM_med = array_rfM_med[pad_bef:-pad_aft]
    assert len(array_rfM_med) == wf_npts


    return array_P_med, array_S_med, array_eqM_med, array_rfM_med


def pred_MedianFilter_evocc(eqocc, rfocc, wf_npts, dt, 
        pred_npts, pred_interval_sec, pad_bef, pad_aft):
    ### 3. Integrate continuous predictions
    wf_n = wf_npts + (pad_bef+pad_aft)
    pred_array_eqocc = [[] for _ in range(wf_n)]
    pred_array_rfocc = [[] for _ in range(wf_n)]

    pred_interval_pt =  int(round(pred_interval_sec/dt))

    init_pt = 0
    for i in range(len(eqocc)):
        eqmm = np.array_split(eqocc[i].T[0], pred_npts)
        rfmm = np.array_split(rfocc[i].T[0], pred_npts)

        j = 0
        for eqm, rfm in zip(eqmm, rfmm):
            pred_array_eqocc[init_pt+j].append(eqm)
            pred_array_rfocc[init_pt+j].append(rfm)
            j += 1
        init_pt += pred_interval_pt
    
    pred_array_eqocc = np.array(pred_array_eqocc, dtype='object')
    pred_array_rfocc = np.array(pred_array_rfocc, dtype='object')
    # fast revision of bottleneck
    lenM = np.array([len(m) for m in pred_array_rfocc])
    nums = np.unique(lenM)
    array_eqocc_med = np.zeros(wf_n)
    array_rfocc_med = np.zeros(wf_n)
    for k in nums:
        num_idx = np.where(lenM==k)[0]
        array_eqocc_med[num_idx] = \
            np.median(np.hstack(
                np.take(pred_array_eqocc, num_idx)), axis=0)
        array_rfocc_med[num_idx] = \
            np.median(np.hstack(
                np.take(pred_array_rfocc, num_idx)), axis=0)

    array_eqocc_med = array_eqocc_med[pad_bef:-pad_aft]
    array_rfocc_med = array_rfocc_med[pad_bef:-pad_aft]
    assert len(array_rfocc_med) == wf_npts


    return array_eqocc_med, array_rfocc_med

def conti_standard_array_fast(array, pred_npts, pred_interval_sec, dt):
    '''
    input: 
    array: np.ndarray (raw_data)
    pred_npts
    pred_interval_sec
    
    output:
    wf_slices
    wf_start_utc
    '''
    raw_n = len(array[0].data)
    pred_rate = int(pred_interval_sec/dt)
    full_len = int(pred_npts + pred_rate*\
        np.ceil(raw_n-pred_npts)/pred_rate)
    n_marching_win = int((full_len - pred_npts)/pred_rate)+1
    is_full_pred = (np.divmod((full_len - pred_npts + 1), pred_rate)[1] == 0)
    #last_null_pt = 0
    #if not is_full_pred:
    #    last_null_pt = pred_rate*(n_marching_win-1)+pred_npts
    last_null_pt = pred_rate*(n_marching_win-1)+pred_npts

    wf_slices = []
    n_marching_win_total = n_marching_win 

    for w in range(len(array)):
        wf_ = np.array(
            [deepcopy(array[w])[pred_rate*i:pred_rate*i+pred_npts]
                for i in range(n_marching_win)])
        if last_null_pt > 0:
            wf_last = deepcopy(array[w])[raw_n-pred_npts:raw_n]   
            wf_ = np.vstack([wf_, wf_last])
            n_marching_win_total = n_marching_win + 1

        wf_ -= np.repeat(np.mean(wf_, axis=1),
            pred_npts, axis=0).reshape(n_marching_win_total, pred_npts)
        wf_ /= np.repeat(np.std(wf_, axis=1),
            pred_npts, axis=0).reshape(n_marching_win_total, pred_npts)

        # check infinity & nan
        i_nan = np.where(np.isnan(wf_))[0]
        i_inf = np.where(np.isinf(wf_))[0]
        if len(i_nan) > 0:
            wf_[i_nan] = np.zeros(len(i_nan))
        if len(i_inf) > 0:
            wf_[i_inf] = np.zeros(len(i_inf))
        assert np.any(np.isinf(wf_))==False
        assert np.any(np.isnan(wf_))==False

        wf_slices.append(wf_)

    wf_slices = np.stack([wf_slices[0], wf_slices[1], wf_slices[2]], -1)

    return np.array(wf_slices), n_marching_win_total, last_null_pt

def conti_pred_rebuild(picks, masks, n_marching_win_total,
        last_null_pt, tfr_length, pred_interval_sec, dt):

    pred_npts = picks.shape[1]
    pred_array_mask =[[] for _ in range(tfr_length)]
    pred_interval_pt =  int(round(pred_interval_sec/dt))

    # bottleneck
    init_pt = 0
    for i in range(len(picks)):
        mm = np.array_split(masks[i].T[0], pred_npts)
        
        # if the marching window cannot cover the whole waveform
        if np.logical_and(last_null_pt > 0, i==len(picks)-1):
            init_pt = tfr_length - pred_npts
        
        j = 0
        for m in mm:
            pred_array_mask[init_pt+j].append(m)
            j += 1
        init_pt += pred_interval_pt
        
    pred_array_mask = np.array(pred_array_mask, dtype='object')

    lenM = np.array([len(m) for m in pred_array_M])
    nums = np.unique(lenP)
    M_med = np.zeros(tfr_length)
    for k in nums:
        num_idx = np.where(lenM==k)[0]
        M_med[num_idx] = np.median(
            np.hstack(np.take(pred_array_mask, num_idx)), axis=0)
    return M_med
    
def IsRightTrigger(gt_range, trigger_mask):
    '''
    gt_range: trigger range of ground truth
    trigger_mask: trigger ranges of prediction masks


    '''
    # 1 for right Trigger; 0 for wrong Trigger/No trigger
    def check_trigger(gt_range, trg_range):
        if trg_range[0]>gt_range[0] and trg_range[1]<gt_range[1]:
            return 1
        else:
            return 0
    T = np.sum([check_trigger(gt_range, t) for t in trigger_mask])
    if T > 1:
        T = 0
    return T

def trg_peak_value(pred_func, trg_st_thre=0.1, trg_end_thre=0.1):
    '''
    Check the maximum value of predictions trigger function
    '''
    trg_func = trigger_onset(pred_func, trg_st_thre, trg_end_thre)
    if len(trg_func) == 0:
        max_pk_value = 0
    else:
        peak_values = []
        for trg in trg_func:
            peak_value = np.max(pred_func[trg])
            if peak_value >= 0.1:
                peak_values.append(peak_value)
        max_pk_value = np.max(peak_values)
    return max_pk_value


def pick_peaks(prediction, labeled_phase, sac_dt=None,
                     search_win=1, peak_value_min=0.01):
    '''
    search for potential pick
    
    parameters
    ----
    prediction: predicted functions
    labeled_phase: the timing of labeled phase
    sac_dt: delta of sac 
    search_win: time window (sec) for searching 
    local maximum near labeled phases 
    '''
    try:
        tphase = int(round(labeled_phase/sac_dt))
        search_range = [tphase-int(search_win/sac_dt), 
                        tphase+int(search_win/sac_dt)]
        peaks, values = find_peaks(prediction, height=peak_value_min)

        in_search = [np.logical_and(v>search_range[0], 
                        v<search_range[1]) for v in peaks]
        _peaks = peaks[in_search]
        _values = values['peak_heights'][in_search]
        return _peaks[np.argmax(_values)]*sac_dt, \
                _values[np.argmax(_values)]
    except ValueError:
        return -999, -999

class RockFall_STMF:
    def __init__(
        self,
        model=None,
        dt=0.01,
        pred_npts=6000, 
        pred_interval_sec=2,
        station_num=4,
        highpass=None
        ):

        self.model = model
        self.dt = dt
        self.pred_npts = pred_npts
        self.pred_interval_sec = pred_interval_sec
        self.station_num=station_num
        self.highpass=highpass

        if model == None:
            AssertionError("The Phase picker model should be defined!")

    def predict(self, wf_list=None, single_output=True):
        from time import time
        if len(wf_list[0][0]) < self.pred_npts:
            AssertionError(
                f"Data should be longer than {self.pred_npts} points.")
        ## store continuous data into array according to prediction interval

            #wf_slices, spectrogram, pad_bef, pad_aft = conti_standard_wf_fast(
            #    wf_list[c], 
            #    pred_npts=self.pred_npts, 
            #    pred_interval_sec=self.pred_interval_sec, dt=self.dt)
        net_wf_slices = []
        net_spectrogram = []
        net_pad_bef = []
        net_pad_aft = []            
        for cc in range(self.station_num):
            wf_slices, spectrogram, pad_bef, pad_aft = \
                conti_standard_wf_fast(
                    wf_list[cc].T, 
                    pred_npts=self.pred_npts, 
                    pred_interval_sec=self.pred_interval_sec, dt=self.dt
            )  

            net_wf_slices.append(wf_slices)
            net_spectrogram.append(spectrogram)
            net_pad_bef.append(pad_bef)
            net_pad_aft.append(pad_aft)
        net_wf_slices = np.stack(net_wf_slices, axis=1)
        net_spectrogram = np.stack(net_spectrogram, axis=1)
        net_pad_bef = np.array(net_pad_bef)
        net_pad_aft = np.array(net_pad_aft)

        ## make prediction
        t1 = time()
        rfocc, eqocc, eqpick, eqmask, rfmask = self.model.predict(
            [net_wf_slices, net_spectrogram]
        )
        print(f"Prediction making: {time()-t1:.2f} secs")
        ## apply median filter to sliding predictions
        wf_npts = len(wf_list[0].T[0])

        if single_output:
            # (1) waveform
            c_P_med = []
            c_S_med = []
            c_eqM_med = []
            c_rfM_med = []
            for i in range(len(wf_list)):
                wf_npts = len(wf_list[i].T[0])
                array_P_med, array_S_med, array_eqM_med, array_rfM_med = \
                    pred_MedianFilter(
                        eqpick=eqpick[:, i, ...], eqmask=eqmask[:, i, ...],
                        rfmask=rfmask[:, i, ...], wf_npts=wf_npts, 
                        dt=self.dt, pred_npts=self.pred_npts, 
                        pred_interval_sec=self.pred_interval_sec,
                        pad_bef=pad_bef, pad_aft=pad_aft
                    )
                
                # replace nan by 0
                find_nan = np.where(np.isnan(array_P_med))[0]
                array_P_med[find_nan] = np.zeros(len(find_nan))
                array_S_med[find_nan] = np.zeros(len(find_nan))
                array_eqM_med[find_nan] = np.zeros(len(find_nan))
                array_rfM_med[find_nan] = np.zeros(len(find_nan))
                # replace inf by 0
                find_inf = np.where(np.isnan(array_P_med))[0]
                array_P_med[find_inf] = np.zeros(len(find_inf))
                array_S_med[find_inf] = np.zeros(len(find_inf))
                array_eqM_med[find_inf] = np.zeros(len(find_inf))
                array_rfM_med[find_inf] = np.zeros(len(find_inf))

                c_P_med.append(array_P_med)
                c_S_med.append(array_S_med)
                c_eqM_med.append(array_eqM_med)
                c_rfM_med.append(array_rfM_med)

            c_P_med = np.array(c_P_med)
            c_S_med = np.array(c_S_med)
            c_eqM_med = np.array(c_eqM_med)
            c_rfM_med = np.array(c_rfM_med)

        # (2) event occurrence
        med_eqocc, med_rfocc = pred_MedianFilter_evocc(
                    eqocc=eqocc, rfocc=rfocc,
                    wf_npts=wf_npts, 
                    dt=self.dt, pred_npts=self.pred_npts, 
                    pred_interval_sec=self.pred_interval_sec,
                    pad_bef=pad_bef, pad_aft=pad_aft
                )
        # replace nan by 0
        find_nan = np.where(np.isnan(med_rfocc))[0]
        med_eqocc[find_nan] = np.zeros(len(find_nan))
        med_rfocc[find_nan] = np.zeros(len(find_nan))
        # replace inf by 0
        find_inf = np.where(np.isnan(med_rfocc))[0]
        med_eqocc[find_inf] = np.zeros(len(find_inf))
        med_rfocc[find_inf] = np.zeros(len(find_inf))


        if single_output:
            return med_rfocc, med_eqocc, c_P_med, c_S_med, c_eqM_med, c_rfM_med
        else:
            return med_rfocc, med_eqocc
            
if __name__ == '__main__':
    pass
