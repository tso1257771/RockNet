import numpy as np
import scipy.signal as ss
from scipy.signal import tukey
from scipy.signal import find_peaks
from obspy import read
from obspy.signal import filter
from obspy.signal.filter import envelope
from obspy.signal.trigger import trigger_onset
from copy import deepcopy

def gen_tar_func(data_length, point, mask_window):
    '''
    data_length: target function length
    point: point of phase arrival
    mask_window: length of mask, must be even number
                 (mask_window//2+1+mask_window//2)
    '''
    target = np.zeros(data_length)
    half_win = mask_window//2
    gaus = np.exp(-(
        np.arange(-half_win, half_win+1))**2 / (2*(half_win//2)**2))
    #print(gaus.std())
    gaus_first_half = gaus[:mask_window//2]
    gaus_second_half = gaus[mask_window//2+1:]
    target[point] = gaus.max()
    #print(gaus.max())
    if point < half_win:
        reduce_pts = half_win-point
        start_pt = 0
        gaus_first_half = gaus_first_half[reduce_pts:]
    else:
        start_pt = point-half_win
    target[start_pt:point] = gaus_first_half
    target[point+1:point+half_win+1] = \
        gaus_second_half[:len(target[point+1:point+half_win+1])]

    return target

def gen_tar_func_triangle(data_length, point, mask_window):
    '''
    data_length: target function length
    point: point of phase arrival
    mask_window: length of mask, must be even number
                 (mask_window//2+1+mask_window//2)
    '''
    target = np.zeros(data_length)

    tri_first_half = np.linspace(0, 1, mask_window//2)[:-1]
    tri_second_half = tri_first_half[::-1]

    target[point] = 1
    # triangle first half
    if (point-len(tri_first_half)) >= 0:
        target[point-len(tri_first_half):point] = tri_first_half
    elif (point-len(tri_first_half)) < 0:
        reduce_pt = len(tri_first_half) - point + 1
        target[0:reduce_pt] = tri_first_half[-reduce_pt:]

    # triangle second half
    if (len(tri_second_half) + point) > data_length:
        reduce_pt = (len(tri_second_half) + point) - data_length
        #target[point+1:point+reduce_pt+1] = \
        #    tri_second_half[:-(len(tri_second_half) - reduce_pt)]
        target[-reduce_pt:] = \
            tri_second_half[:-(len(tri_second_half) - reduce_pt)]      
    elif (len(tri_second_half) + point) == data_length:   
        target[point+1:point+len(tri_second_half)] = tri_second_half[:-1]             
    elif (len(tri_second_half) + point) < data_length:
        target[point+1:point+len(tri_second_half)+1] = tri_second_half 
    return target

def BP_envelope_eq_info(glob_list, freqmin=1, freqmax=20,
        p_hdr='t5', s_hdr='t6', std_pt_bef_P=0.5, mva_win_sec=3,
        env_end_ratio=0.8):
    '''Define an earthquake event using filtered envelope function
    and its moving average.
    # --------- input
    * read_list: list of event files
    * freqmin / freqmax: corner frequency for band-pass filtering
    * p_hdr / s_hdr: index of labeled P/S phase stored in sac header
    * std_pt_ber_P: standard check point for moving average value
            of the envelope function before labeled P arrival (secs)
    * mva_win_sec: window length for calculating moving average (secs)
    * env_end_ratio: ratio of event end point and standard point on 
            envelope function.
    # -------- output
    * tp_pt / ts_pt 
    * dt
    * st_env
    * env_mva
    * env_standard_pt 
    * end_ev_pt
    '''
    st = read(glob_list)
    st.sort()
    st = st.detrend('demean')
    for s in st:
        s.data /= np.std(s.data)
    st = st.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
    info = st[0].stats
    dt = info.delta
    tp_pt = int(round((info.sac[p_hdr] - info.sac.b)/dt, 2))
    ts_pt = int(round((info.sac[s_hdr] - info.sac.b)/dt, 2))

    env_standard_pt = int(round((
        info.sac.t5 - info.sac.b - std_pt_bef_P)/dt, 2))

    # establish moving average function of filtered envelope function
    arr_shape = len(st), len(st[0].data)
    mva_win = int(mva_win_sec/dt)
    st_env = np.zeros(arr_shape)
    env_mva = np.zeros(arr_shape)
    for ct, p in enumerate(st):
        st_env[ct] = envelope(p.data)
        mva = np.convolve(st_env[0],
            np.ones((mva_win,))/mva_win, mode='valid')
        env_mva[ct][-len(mva):] = mva
    # find event end point across channels
    end_ev_pts = np.zeros(2)
    for i in range(len(st)-1):
        end_v_search = np.where(
            env_mva[i] <= env_mva[i][env_standard_pt]/env_end_ratio )[0]
        end_ev_pts[i] = end_v_search[ 
            np.where(end_v_search-ts_pt > 0)[0][0] ]
    end_ev_pt = int(np.mean(end_ev_pts))

    return tp_pt, ts_pt, dt, st_env, env_mva, env_standard_pt, end_ev_pt

def assign_slice_window(p_s_residual, data_length, 
                        avail_bef_P, avail_aft_S, dt):
    """
    `p_s_residual`: |P_arrival - S_arrival|
    `data_length`: total length of sliced waveform
    `avail_bef_P`: available dataspace before P phase
    `avail_aft_S`: available dataspace ater S phase

    Conditioning
    -P_prewin >= avail_bef_P
    -S_prewin = P_prewin + p_s_residual
    -(S_prewin + avail_aft_S ) < data_length

    P_prewin: length of time window before P arrival
    return P_prewin
    
    """
    avail_bef_P/=dt
    avail_aft_S/=dt

    P_avail_space = np.arange(avail_bef_P, 
                (data_length - p_s_residual - avail_aft_S), 1)
    P_prewin = np.random.choice(P_avail_space)
    return P_prewin

def snr_pt(tr, pt, mode='std',
            snr_win=5, highpass=None):
    """
    Calculate snr
    tr: sac trace
    pt: utcdatetime object
    """
    tr_s = tr.copy()
    tr_n = tr.copy()
    if highpass:
        tr = tr.filter(
            'highpass', freq=highpass).\
            taper(max_percentage=0.1, max_length=0.1)

    if mode.lower() == 'std':
        tr_noise = np.std(tr_n.slice(pt-snr_win, pt).data)
        tr_pt = np.std(tr_s.slice(pt, pt+snr_win).data)
        snr = tr_pt/tr_noise

    elif mode.lower() == 'sqrt':
        tr_noise = np.sqrt(np.square(
            tr_n.slice(pt-snr_win, pt).data).sum())
        tr_pt = np.sqrt(np.square(
            tr_s.slice(pt, pt+snr_win).data).sum())
        snr = tr_pt/tr_noise

    elif mode.lower() == 'db':
        tr_noise = np.std(tr_n.slice(pt-snr_win, pt).data)
        tr_pt = np.std(tr_s.slice(pt, pt+snr_win).data) 
        snr = 10*np.log10(tr_pt/tr_noise)
    return snr

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

def snr_pt_v3(tr_vertical, tr_horizontal, pt_p, pt_s, mode='std',
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
            'highpass', freq=highpass).taper(
                max_percentage=0.1, max_length=0.1)
        tr_horizontal = tr_horizontal.filter(
            'highpass', freq=highpass).taper(
                max_percentage=0.1, max_length=0.1)
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

def mosaic_Psnr_npts(z_trc, gt_label_P, P_snr_win=3, dt=0.01,
    data_length=2001, hpfreq=2, mode='sqrt'):
    if hpfreq:
        _hpZtrc = tukey(data_length, alpha=0.1)*\
            (filter.highpass(z_trc, freq=hpfreq, df=1/dt))
    else:
        _hpZtrc = z_trc

    _P_snr_win = int(P_snr_win/0.01)
    Psnrs = []
    for sufP in range(len(gt_label_P)):
        if gt_label_P[sufP] < _P_snr_win:
            snr_win = gt_label_P[sufP]-1
        elif gt_label_P[sufP] + _P_snr_win > data_length:
            snr_win = data_length - gt_label_P[sufP] - 1
        else:
            snr_win = _P_snr_win

        sig = _hpZtrc[gt_label_P[sufP]:gt_label_P[sufP]+snr_win]
        nz = _hpZtrc[gt_label_P[sufP]-snr_win:gt_label_P[sufP]]
        if mode.lower() == 'std':
            tr_noise = np.std(nz)
            tr_pt = np.std(sig)
        elif mode.lower() == 'sqrt':
            tr_noise = np.sqrt(np.square(nz).sum())
            tr_pt = np.sqrt(np.square(sig).sum())
        Psnrs.append(tr_pt/tr_noise)      
    return  Psnrs

def MWA_suffix_Psnr(trc_mosaic, tp_to_ori_pt, snr_win=0.5, dt=0.01, 
                    hpfreq=2, mode='std'):
    '''Calculate SNR of suffix P on merged waveforms
    '''
    suffix_Psnrs = []
    data_npts = len(trc_mosaic[0])
    for sufP in range(len(tp_to_ori_pt[1:])):
        _hpZtrc = tukey(data_npts, alpha=0.1)*\
                    (filter.highpass(trc_mosaic[2], 
                     freq=hpfreq, df=1/dt))
        sufP_pt = tp_to_ori_pt[sufP+1]        
        if sufP_pt >= data_npts:
            suffix_Psnrs.append(999)
        else:
            sig = _hpZtrc[sufP_pt:int(sufP_pt+snr_win/dt)]
            nz = _hpZtrc[int(sufP_pt-snr_win/dt):sufP_pt]
            if mode.lower() == 'std':
                tr_noise = np.std(nz)
                tr_pt = np.std(sig)
            elif mode.lower() == 'sqrt':
                tr_noise = np.sqrt(nz.sum())
                tr_pt = np.sqrt(sig.sum())
            suffix_Psnrs.append(tr_pt/tr_noise)
    return np.array(suffix_Psnrs)

def MWA_joint_Zsnr(trc_mosaic, cumsum_npts, snr_win=1, dt=0.01, 
                    hpfreq=2, mode='std'):
    '''Calculate SNR of waveform joint on vertical component
    '''
    joint_snrs = []
    data_npts = len(trc_mosaic[0])
    for jt in range(len(cumsum_npts[:-1])):
        _hpZtrc = tukey(data_npts, alpha=0.1)*(filter.highpass(
            trc_mosaic[2], freq=hpfreq, df=1/dt))
        jt_pt = cumsum_npts[jt]
        sig = _hpZtrc[jt_pt:int(jt_pt+snr_win/dt)]
        nz = _hpZtrc[int(jt_pt-snr_win/dt):jt_pt]
        if mode.lower() == 'std':
            tr_noise = np.std(nz)
            tr_pt = np.std(sig)
        elif mode.lower() == 'sqrt':
            tr_noise = np.sqrt(nz.sum())
            tr_pt = np.sqrt(sig.sum())
        joint_snrs.append(tr_pt/tr_noise)
    return np.array(joint_snrs)

def MWA_joint_ENZsnr(trc_mosaic, joint_pt, snr_win=1, dt=0.01, 
                    hpfreq=2, mode='std'):
    '''Calculate SNR of waveform joint on vertical component
    '''
    chn_joint_snrs = []
    for ms in trc_mosaic:
        joint_snrs = []
        data_npts = len(trc_mosaic[0])
        for jt in range(len(joint_pt)):
            _hptrc = tukey(data_npts, alpha=0.1)*\
                (filter.highpass(ms, freq=hpfreq, df=1/dt))
            jt_pt = joint_pt[jt]
            sig = _hptrc[jt_pt:int(jt_pt+snr_win/dt)]
            nz = _hptrc[int(jt_pt-snr_win/dt):jt_pt]
            if mode.lower() == 'std':
                tr_noise = np.std(nz)
                tr_pt = np.std(sig)
            elif mode.lower() == 'sqrt':
                tr_noise = np.sqrt(nz.sum())
                tr_pt = np.sqrt(sig.sum())
            joint_snrs.append(tr_pt/tr_noise)
        chn_joint_snrs.append(joint_snrs)
    return np.array(chn_joint_snrs)

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
            s.data[np.isinf(s.data)] = 0
        if np.isnan(s.data).any():
            s.data[np.isnan(s.data)] = 0 
    return st

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
                    arr=wf[w].data, obj=len(wf[w].data), 
                    values=np.full(len(wf[w].data), wf[w].data[-1])
                    )
        elif append_npts < 0:
            wf[w].data = wf[w].data[:max_n]
    return wf

def conti_standard_wf_fast(wf, pred_npts, pred_interval_sec, dt):
    '''
    input: 
    wf: obspy.stream object (raw_data)
    pred_npts
    pred_interval_sec
    
    output:
    wf_slices
    wf_start_utc
    '''
    raw_n = len(wf[0].data)
    pred_rate = int(pred_interval_sec/dt)
    full_len = int(pred_npts + pred_rate*\
        np.ceil(raw_n-pred_npts)/pred_rate)
    n_marching_win = int((full_len - pred_npts)/pred_rate)+1
    n_padded = full_len - raw_n

    wf = sac_len_complement(wf.copy(), max_length=full_len)

    wf_E = np.array([deepcopy(
        wf[0].data[pred_rate*i:pred_rate*i+pred_npts])
        for i in range(n_marching_win)])
    wf_E -= np.repeat(np.mean(wf_E, axis=1),
         pred_npts, axis=0).reshape(n_marching_win, pred_npts)
    wf_E /= np.repeat(np.std(wf_E, axis=1),
         pred_npts, axis=0).reshape(n_marching_win, pred_npts)

    wf_N = np.array([deepcopy(
        wf[1].data[pred_rate*i:pred_rate*i+pred_npts])
        for i in range(n_marching_win)])
    wf_N -= np.repeat(np.mean(wf_N, axis=1),
         pred_npts, axis=0).reshape(n_marching_win, pred_npts)
    wf_N /= np.repeat(np.std(wf_N, axis=1),
         pred_npts, axis=0).reshape(n_marching_win, pred_npts)

    wf_Z = np.array([deepcopy(
        wf[1].data[pred_rate*i:pred_rate*i+pred_npts])
        for i in range(n_marching_win)])
    wf_Z -= np.repeat(np.mean(wf_Z, axis=1),
         pred_npts, axis=0).reshape(n_marching_win, pred_npts)
    wf_Z /= np.repeat(np.std(wf_Z, axis=1),
         pred_npts, axis=0).reshape(n_marching_win, pred_npts)

    # deprecated version
    #wf_Z = np.array([deepcopy(wf[2].data[pred_rate*i:pred_rate*i+pred_npts]) 
    #    for i in range(n_marching_win)]) 
    #wf_Z_dm = np.array([i-np.mean(i) for i in wf_Z])
    #wf_Z_norm = np.array([i/np.std(i) for i in wf_Z_dm])

    wf_slices = np.stack([wf_E, wf_N, wf_Z], -1)
    return np.array(wf_slices), n_padded

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
    #is_full_pred = (np.divmod((full_len - pred_npts + 1), pred_rate)[1] == 0)
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
        wf_slices.append(wf_)

    wf_slices = np.stack([wf_slices[0], wf_slices[1], wf_slices[2]], -1)

    return np.array(wf_slices), n_marching_win_total, last_null_pt

def conti_pred_rebuild(picks, masks, n_marching_win_total,
        last_null_pt, tfr_length, pred_interval_sec, dt):

    pred_npts = picks.shape[1]
    pred_array_P = [[] for _ in range(tfr_length)]
    pred_array_S = [[] for _ in range(tfr_length)]
    pred_array_mask =[[] for _ in range(tfr_length)]
    pred_interval_pt =  int(round(pred_interval_sec/dt))

    # bottleneck
    init_pt = 0
    for i in range(len(picks)):
        pp = np.array_split(picks[i].T[0], pred_npts)
        ss = np.array_split(picks[i].T[1], pred_npts)
        mm = np.array_split(masks[i].T[0], pred_npts)
        
        # if the marching window cannot cover the whole waveform
        if np.logical_and(last_null_pt > 0, i==len(picks)-1):
            init_pt = tfr_length - pred_npts
        
        j = 0
        for p, s, m in zip(pp, ss, mm):
            pred_array_P[init_pt+j].append(p)
            pred_array_S[init_pt+j].append(s)
            pred_array_mask[init_pt+j].append(m)
            j += 1
        init_pt += pred_interval_pt
        
    pred_array_P = np.array(pred_array_P, dtype='object')
    pred_array_S = np.array(pred_array_S, dtype='object')
    pred_array_mask = np.array(pred_array_mask, dtype='object')

    lenP = np.array([len(p) for p in pred_array_P])
    nums = np.unique(lenP)
    P_med = np.zeros(tfr_length)
    S_med = np.zeros(tfr_length)
    M_med = np.zeros(tfr_length)
    for k in nums:
        num_idx = np.where(lenP==k)[0]
        P_med[num_idx] = np.median(
            np.hstack(np.take(pred_array_P, num_idx)), axis=0)
        S_med[num_idx] = np.median(
            np.hstack(np.take(pred_array_S, num_idx)), axis=0)
        M_med[num_idx] = np.median(
            np.hstack(np.take(pred_array_mask, num_idx)), axis=0)  
    return P_med, S_med, M_med

def stream_stft(stream, fs, nperseg, nfft):
    '''compute stft for time series data
    '''
    def mtc_standardize(mtc):
        mtc = mtc - np.mean(mtc)
        mtc = mtc/np.std(mtc)
        return mtc

    FT_data_real = []
    FT_data_imag = []
    scale_FT = []
    stream = stream.detrend('demean')
    for i in range(len(stream)):
        scale_factor = {}
        f, t, tmp_FT = ss.stft(stream[i].data, 
            fs=fs, nperseg=nperseg, nfft=nfft, boundary='zeros')
        tmp_FT_real, tmp_FT_imag = tmp_FT.real, tmp_FT.imag

        scale_factor['real_mean'] =  np.mean(tmp_FT_real)
        scale_factor['real_std'] =  np.std(tmp_FT_real)
        scale_factor['imag_mean'] =  np.mean(tmp_FT_imag)
        scale_factor['imag_std'] =  np.std(tmp_FT_imag)        

        FT_data_real.append(mtc_standardize(tmp_FT_real))
        FT_data_imag.append(mtc_standardize(tmp_FT_imag))
        scale_FT.append(scale_factor)

    FT_data_real = np.stack(FT_data_real, axis=-1)
    FT_data_imag = np.stack(FT_data_imag, axis=-1)
    scale_FT = np.stack(scale_FT, axis=-1)
    return f, t, FT_data_real, FT_data_imag, scale_FT


def pad_stream(trc, data_length, pad_range,
        max_pad_slices=4, random_chn=True, max_pad_chn_num=3):
    '''
    Randomly pad the waveform with zero values on channels

    '''
    # 0 for padding with zero; 1 for no padding
    if random_chn:
        pad_chn = np.array([np.random.randint(2) for i in range(3)])
        pad_chn_idx = np.where(pad_chn==1)[0]
        if pad_chn_idx.sum()==0:
            pad_chn_idx = np.array([0, 1, 2])
    else:
        pad_chn_idx = np.array([0, 1, 2])
    if len(pad_chn_idx) >= max_pad_chn_num:
        pad_chn_idx = np.random.permutation(
            pad_chn_idx)[:max_pad_chn_num-1]

    zero_pad = np.random.randint(
        pad_range[0], pad_range[1])
    max_pad_seq_num = np.random.randint(max_pad_slices)+1
    pad_len = np.random.multinomial(zero_pad, 
        np.ones(max_pad_seq_num)/max_pad_seq_num)

    pad_num = [0, -30, 30, 0]
    pad_obj = pad_num[np.random.choice(3)]
    for ins in range(len(pad_len)):
        max_idx = data_length - pad_len[ins]
        insert_idx = np.random.randint(max_idx)
        for ch in pad_chn_idx:
            trc[ch].data[insert_idx:(insert_idx+pad_len[ins])] = \
                np.full(pad_len[ins], pad_obj)
    trc = stream_standardize(trc, data_length)
    return trc

if __name__ == '__main__':
    pass
