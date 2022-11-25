import os
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Helvetica'
from obspy import read
from glob import glob
from .data_utils import gen_tar_func

class Mosaic_waveform:
    def __init__(self):
        pass

    def _mosaic_basic_info(self, data_path, ev_pair, base_wf_sec=20, times_residual_aftS=3,
                        secs_bef_P=1, use_aicP=True, use_aicS=True):
        '''
        1. data_path            : `str`, path for read waveform
        2. ev_pair              : `list`, lists of waveform for concatenation
        3. base_wf_sec          : `float`,standard input length of seismic data into U-Net like models (seconds)
        4. times_residual_aftS  : `int`, base magnification of |S-P| residual left after S arrivals,
                                    e.g. |S-P|=3, at least 3*3=9 seconds were left after S arrivals.
        5. secs_bef_P           : `float`, least length of data left before labeled P arrivals.
        6. use_aicP, use_aicS   : `float`, whether to use picks picked by AIC picker
        (stored in SACHEADER.sac.t5, SACHEADER.sac.t6)
        ---------------------------------------------------------------
        return decimation, p_utc, s_utc, trc_pairs, init_stt, init_ent
        1. decimation   : `int`; decimation magnification for concatenated waveform
        2. p_utc        : `list`; UTCDateTime object of events P arrivals
        3. s_utc        : `list`; UTCDateTime object of events S arrivals
        4. trc_pairs    : `list`; ObsPy.SACTrace object
        5. init_stt     : `list`; UTCDateTime objects of initial starttime of each set of waveform
        6. init_ent     : `list`; UTCDateTime objects of initial endtime of each set of waveform
            $init_stt and $init_ent ensures enough information be included in the sliced waveform,
            determined by specified $times_residual_aftS and $secs_bef_P
        '''
        #secs_bef_P=1
        #times_residual_aftS=3
        #base_wf_sec = 20
        #use_aicP = True
        #use_aicS = True
        # calculate residual to fill between events
        p_utc = []
        residual_ps = []
        trc_pairs = []
        for t in range(len(ev_pair)):
            subdir = '/'.join(ev_pair[t].split('/')[7:10])
            m_type = '.'.join(ev_pair[t].split('/')[10].split('.')[1:3])
            trc = read(os.path.join(data_path, subdir, f'*.{m_type}.*.sac'))
            info = trc[0].stats
            if use_aicP: tp = info.starttime-info.sac.b+info.sac.t5
            elif not use_aicP: tp = info.starttime-info.sac.b+info.sac.t1
            if use_aicP: ts = info.starttime-info.sac.b+info.sac.t6
            elif not use_aicS: tp = info.starttime-info.sac.b+info.sac.t2
            residual = ts-tp
            p_utc.append(tp); residual_ps.append(residual); trc_pairs.append(trc)

        p_utc = np.array(p_utc)
        s_utc = p_utc + residual_ps
        secs_aft_P = np.array(residual_ps)*(1+times_residual_aftS)
        base_period = secs_aft_P.sum()+len(residual_ps)*secs_bef_P

        init_stt = np.array(p_utc) - secs_bef_P
        init_ent = np.array(p_utc) + secs_aft_P
        decimation = int(np.divmod(base_period, base_wf_sec)[0]) + 2
        return decimation, p_utc, s_utc, trc_pairs, init_stt, init_ent

    def _mosaic_segment_info(self, base_wf_sec, decimation, p_utc, s_utc, trc_pairs, init_stt, init_ent):
        '''
        1. base_wf_sec  : `float`;length of mosaic waveform in seconds
        2. decimation   : `int`; decimation magnification for concatenated waveform
        3. p_utc        : `list`; UTCDateTime object of events P arrivals
        4. s_utc        : `list`; UTCDateTime object of events S arrivals
        5. trc_pairs    : `list`; ObsPy.SACTrace object
        6. init_stt     : `list`; UTCDateTime objects of initial starttime of each set of waveform
        7. init_ent     : `list`; UTCDateTime objects of initial endtime of each set of waveform
            $init_stt and $init_ent ensures enough information be included in the sliced waveform,
            determined by specified $times_residual_aftS and $secs_bef_P
        ---------------------------------------------------------------
        return slice_trc, tp_to_ori_pt, ts_to_ori_pt, cumsum_npts, data_npts
        1. slice_trc    : `list`; Sliced and decimated waveform of event pairs for concatenation
        2. tp_to_ori_pt : `list`; Points of labeled P arrivals
        3. ts_to_ori_pt : `list`; Points of labeled S arrivals
        4. cumsum_npts  : `list`; Cumulated npts of event pairs after concatenation
        5. data_npts    : `int`; npts for full length of mosaic waveform
        6. raw_dt       : `float`, Delta of raw sliced waveform, which must be identical with and passed to mosaic waveform
        '''
        ## assert ev_pairs with identical sampling rate and define npts for mosaic waveform
        raw_dt = np.unique([Trc[0].stats.delta for Trc in trc_pairs])
        assert len(raw_dt) == 1
        data_npts = int(base_wf_sec/raw_dt[0]+1)

        ## randomly assign available data space (seconds) before and after every init_stt and init_ent
        total_len = decimation*base_wf_sec #seconds
        _res = total_len-np.sum(init_ent-init_stt)
        assign_res = np.random.dirichlet(np.ones(2*len(trc_pairs)))*_res

        ## set up starttime and endtime point for each trace
        slice_idx = [(init_stt[i]-assign_res[i*2], init_ent[i]+assign_res[i*2+1])
                            for i in range(len(assign_res)//2)]
        #base_trc = trc_pairs[0].copy().slice(slice_idx[0][0], slice_idx[0][0]+total_len).decimate(decimation)
        slice_trc = [trc_pairs[i].copy().slice(slice_idx[i][0], slice_idx[i][1]).detrend('demean').decimate(decimation)
                            for i in range(len(slice_idx))]
        # normalize waveform previously
        #for s1 in range(len(slice_trc)):
        #    for s2 in range(len(slice_trc[s1])):
        #        slice_trc[s1][s2].data = slice_trc[s1][s2].data - np.mean(slice_trc[s1][s2].data)
        #        slice_trc[s1][s2].data = slice_trc[s1][s2].data / np.std(slice_trc[s1][s2].data)


        cumsum_npts = np.cumsum([len(slice_trc[i][0]) for i in range(len(slice_trc))])
        appended_npts = cumsum_npts[:-1]

        ## mapping P/S labels to down-sampled time points
        dt = slice_trc[0][0].stats.delta
        tp_to_ori = p_utc-np.array([s[0] for s in slice_idx])
        ts_to_ori = s_utc-np.array([s[0] for s in slice_idx])
        tp_to_ori_pt = np.array([int(round(f)) for f in tp_to_ori/dt])
        ts_to_ori_pt = np.array([int(round(f)) for f in ts_to_ori/dt])
        tp_to_ori_pt[1:]+=appended_npts
        ts_to_ori_pt[1:]+=appended_npts

        return slice_trc, tp_to_ori_pt, ts_to_ori_pt, cumsum_npts, data_npts, raw_dt


    def gen_mosaic_wf(self, slice_trc, tp_to_ori_pt, ts_to_ori_pt, cumsum_npts,
                        data_npts, raw_dt=0.01, err_win_p=0.4, err_win_s=0.7):
        '''
        1. slice_trc    : `list`; Sliced and decimated waveform of event pairs for concatenation
        2. tp_to_ori_pt : `list`; Points of labeled P arrivals
        3. ts_to_ori_pt : `list`; Points of labeled S arrivals
        4. cumsum_npts  : `list`; Cumulated npts after each concatenation operation
        5. data_npts    : `int`; npts for full length of mosaic waveform
        6. raw_dt       : `float`; delta of mosaic waveform, which is identical to that of raw waveforms
        7. err_win_p    : `float`; standard deviation for making P target function (seconds)
        8. err_win_s    : `float`; standard deviation for making S target function (seconds)
        ---------------------------------------------------------------
        return trc_3C, label_psn
        1. trc_3C       : `list`; 3-component mosaic waveform (E-N-Z) in the form of `np.array`
        2. label_psn    : `list`; target function of P, S, and others in the form of `np.array`
        '''
        # estimate max vlue of every component for data compression
        #compress_mag = 5
        #max_v = np.array([np.max(m.data) for i in range(len(slice_trc)) for m in slice_trc[i]])
        #max_v_trc = np.array([max_v[3*i:3*(i+1)] for i in range(len(max_v)//3)])
        # 0 for E, 1 for N, 2 for Z
        #max_ratio = [np.max(max_v_trc[:, i])/ np.min(max_v_trc[:, i]) for i in range(3)]
        #compress_trc = np.argmin(max_v_trc[:, np.argmax(max_ratio)])

        # concatenate waveform
        fake_E = np.zeros(data_npts)
        fake_N = np.zeros(data_npts)
        fake_Z = np.zeros(data_npts)
        base_idx = 0
        for i in range(len(cumsum_npts)):
            end_idx = cumsum_npts[i]
            if end_idx > len(fake_E):
                end_idx = len(fake_E)
            trim_data_E = slice_trc[i][0].data
            trim_data_N = slice_trc[i][1].data
            trim_data_Z = slice_trc[i][2].data                
            # check data length
            neat_length = np.unique([len(trim_data_E.data), len(trim_data_N.data), len(trim_data_Z.data)])
            if len(neat_length) != 1:
                neat_idx = np.min(neat_length)
                trim_data_E = trim_data_E[:neat_idx]
                trim_data_N = trim_data_N[:neat_idx]
                trim_data_Z = trim_data_Z[:neat_idx]

                neat_reduce = np.max(neat_length)-np.min(neat_length)
                end_idx -= neat_reduce

            fake_E[base_idx:end_idx] = trim_data_E[:(end_idx-base_idx)]
            fake_N[base_idx:end_idx] = trim_data_N[:(end_idx-base_idx)]
            fake_Z[base_idx:end_idx] = trim_data_Z[:(end_idx-base_idx)]
            #print(base_idx, end_idx)
            base_idx = cumsum_npts[i]
        fake = [fake_E, fake_N, fake_Z]
        for s3 in range(len(fake)):
            fake[s3] = fake[s3] - np.mean(fake[s3])
            fake[s3] = fake[s3] / np.std(fake[s3])
        trc_mosaic = np.array(fake)

        ## gen_tar_func
        tar_p_data = np.zeros(data_npts)
        tar_s_data = np.zeros(data_npts)

        err_win_p = 0.4
        err_win_s = 0.7
        err_win_npts_p = int(err_win_p/raw_dt)+1
        err_win_npts_s = int(err_win_s/raw_dt)+1

        for n in range(len(tp_to_ori_pt)):
            tar_p_data += gen_tar_func(data_npts, tp_to_ori_pt[n], err_win_npts_p)
            tar_s_data += gen_tar_func(data_npts, ts_to_ori_pt[n], err_win_npts_s)
        tar_nz_data = np.ones(data_npts) - tar_p_data - tar_s_data

        trc_3C = trc_mosaic.T
        label_psn = np.array([tar_p_data, tar_s_data, tar_nz_data]).T
        return trc_3C, label_psn


    def mosaic_wf_plot(self, trc_3C, label_psn, new_tps, new_tss, save=None, show=False):
        wf_data = trc_3C.T
        tar_func = label_psn.T

        x_plot = [wf_data[0], wf_data[1], wf_data[2], tar_func[0], tar_func[1], tar_func[2]]
        label = ["E comp.", "N comp.", "Z comp.", "P prob.", "S prob.", 'Noise prob']
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True, figsize=(8, 8))
        ax=[ax1, ax2, ax3, ax4, ax5, ax6]
        for i in range(6):
            ax[i].plot(x_plot[i], linewidth=1)
            if 2 >= i: 
                for j in range(len(new_tps)):
                    ax[i].axvline(x=new_tps[j], label='Manual picked P', color='k', linewidth=1)
                    ax[i].axvline(x=new_tss[j], label='Manual picked S ', color='r', linewidth=1)  
            ax[i].set_ylabel(label[i])
        ax[0].set_title('Mosaic-concatenated waveform')
        if save:
            plt.savefig(save, dpi=150)
            plt.close()
        if show:
            plt.show()
        return fig

if __name__=='__main__':
    pass