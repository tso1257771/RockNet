import os
import sys
sys.path.append('../')
sys.path.append('../tools/build_model')
import shutil
import tensorflow as tf
import numpy as np
import scipy.signal as ss
import sys
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Helvetica'
import matplotlib.gridspec as gridspec
import scipy.signal as ss
from copy import deepcopy
from glob import glob
from obspy import read, UTCDateTime
from obspy.signal.invsim import corn_freq_2_paz
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from tools.rockfall_net_STMF_fusion import compute_STFT, sac_len_complement
os.environ["CUDA_VISIBLE_DEVICES"] = ""

wfdir = './sac'
predir = './net_pred'
outfig = './fig'
if os.path.exists(outfig):
    shutil.rmtree(outfig)
    os.makedirs(outfig)

jidx = ['2020.088.13', '2020.097.21']
utc_slice = np.array([
        ['2020-03-28T13:41:20', '2020-03-28T13:42:20'],
        ['2020-04-06T21:23:44', '2020-04-06T21:24:44']
    ]
)
# instr. resp.
paz = {'poles':[(-19.781+20.2027j), (-19.781-20.2027j)], 
        'zeros':[0j, 0j], 
        'gain':1815347200.0, 
        'sensitivity':1}
paz_1hz = corn_freq_2_paz(1, damp=0.707)
paz_1hz['sensitivity'] = 1.0

# collect waveform and predictions
stas = ['LH01', 'LH02', 'LH03', 'LH04']
for i in range(len(jidx)):
    stt = UTCDateTime(utc_slice[i][0])
    ent = UTCDateTime(utc_slice[i][1])
    t_sec = int(ent-stt)
    t_npts = int(t_sec*100)

    r_trc = []
    net_spec = []
    sta_p = []
    sta_s = []
    sta_eqmask = []
    sta_rfmask = []
    avail_stas = np.sort([os.path.basename(s).split('.')[1] 
        for s in glob(os.path.join(wfdir, jidx[i], '*.EHZ.*'))])
    for j in range(len(stas)):
        if stas[j] in avail_stas:
            st = read(os.path.join(wfdir, 
                jidx[i], f'*.{stas[j]}.EHE.*.sac'),
                starttime=stt, endtime=ent, nearest_sample=False)
            st += read(os.path.join(wfdir, 
                jidx[i], f'*.{stas[j]}.EHN.*.sac'),
                starttime=stt, endtime=ent, nearest_sample=False)
            st += read(os.path.join(wfdir, 
                jidx[i], f'*.{stas[j]}.EHZ.*.sac'),
                starttime=stt, endtime=ent, nearest_sample=False)

            eqmask = sac_len_complement(
                read(os.path.join(predir, 
                    jidx[i], f'{stas[j]}.{jidx[i]}.sac.eqmask'),
                    starttime=stt, endtime=ent, nearest_sample=False),
                    max_length=t_npts
                )
            rfmask = sac_len_complement(
                read(os.path.join(predir, 
                    jidx[i], f'{stas[j]}.{jidx[i]}.sac.rfmask'),
                    starttime=stt, endtime=ent, nearest_sample=False),
                    max_length=t_npts
                )
            predP = sac_len_complement(
                read(os.path.join(predir, 
                    jidx[i], f'{stas[j]}.{jidx[i]}.sac.P'),
                    starttime=stt, endtime=ent, nearest_sample=False),
                    max_length=t_npts
                )
            predS = sac_len_complement(
                read(os.path.join(predir, 
                    jidx[i], f'{stas[j]}.{jidx[i]}.sac.S'),
                    starttime=stt, endtime=ent, nearest_sample=False),
                    max_length=t_npts
                )                

            r_st = sac_len_complement(deepcopy(st), max_length=t_npts)
            r_st.simulate(paz_remove=paz, paz_simulate=paz_1hz)
            r_st.filter('highpass', freq=5)

            spec = compute_STFT(r_st[2].data)
            r_trc_3C = np.array([s.data[:t_npts] for s in r_st]).T

            net_spec.append(spec)
            r_trc.append(r_trc_3C)
            sta_p.append(predP[0].data)
            sta_s.append(predS[0].data)
            sta_eqmask.append(eqmask[0].data)
            sta_rfmask.append(rfmask[0].data)
        else:
            pseudo_trc = np.random.random((t_npts, 3))
            pseudo_spec = compute_STFT(pseudo_trc.T[2])
            r_trc.append(pseudo_trc)
            net_spec.append(pseudo_spec)
            sta_p.append(np.zeros(t_npts))
            sta_s.append(np.zeros(t_npts))
            sta_eqmask.append(np.zeros(t_npts))
            sta_rfmask.append(np.zeros(t_npts))

    net_spec = np.array(net_spec)
    r_net_trc = np.array(r_trc)
    sta_p = np.array(sta_p)
    sta_s = np.array(sta_s)
    sta_eqmask = np.array(sta_eqmask)
    sta_rfmask = np.array(sta_rfmask)

    f, t, _ = ss.stft(r_net_trc[0].T[2], fs=100, nperseg=20, 
        nfft=100, boundary='zeros') 
    x = np.arange(t_npts)*0.01

    r_trc_E =  np.array([r_net_trc[p].T[0] for p in range(4)])
    r_trc_N =  np.array([r_net_trc[p].T[1] for p in range(4)])
    r_trc_Z =  np.array([r_net_trc[p].T[2] for p in range(4)])
    Z_spec = np.array([i[..., 0]+i[..., 1]*1j for i in net_spec])

    rfocc = sac_len_complement(read(
        os.path.join(predir, 
            jidx[i], f'Luhu.{jidx[i]}.sac.rfocc'),
            starttime=stt, endtime=ent, nearest_sample=False),
            max_length=t_npts
        )[0].data
    eqocc = sac_len_complement(read(
        os.path.join(predir, 
            jidx[i], f'Luhu.{jidx[i]}.sac.eqocc'),
            starttime=stt, endtime=ent, nearest_sample=False),
            max_length=t_npts
        )[0].data

    # plot figures
    ylbl = [
    'E', 'LH01 (1e-6 m/s)\nN', 'Z', '', '',
    'E', 'LH02\nN', 'Z', '', '',
    'E', 'LH03\nN', 'Z', '', '',
    'E\n', 'LH04\nN', 'Z', '', '',
    '', 
    '']
    fig = plt.figure(figsize=(9, 9))
    ax_global = gridspec.GridSpec(22, 1, 
        figure=fig, hspace=0.2, wspace=0.15,
        top=0.96, left=0.11, right=0.95, bottom=0.07)
    ax = [fig.add_subplot(ax_global[j, 0]) for j in range(22)]

    for j in range(22):
        if j in [0, 5, 10, 15]:
            jid = j//4
            # E
            ax[j].plot(x, r_trc_E[jid]/1e-6, linewidth=0.5,
                color='navy', alpha=0.7)
            # N
            ax[j+1].plot(x, r_trc_N[jid]/1e-6, linewidth=0.5,
                color='slategray', alpha=0.7)
            # Z     
            ax[j+2].plot(x, r_trc_Z[jid]/1e-6, linewidth=0.5, 
                color='olive', alpha=0.7)

            ax[j+4].plot(x, sta_eqmask[jid], linewidth=1.5, color='g',
                alpha=0.7,
                label='Earthquake mask')
            ax[j+4].plot(x, sta_rfmask[jid], linewidth=1.5, color='black',
                alpha=0.7,
                label='Rockfall mask')
            ax[j+4].plot(x, sta_p[jid], linewidth=1.5, color='b', label='P',
                alpha=0.7)
            ax[j+4].plot(x, sta_s[jid], linewidth=1.5, color='r',label='S',
                alpha=0.5)
            ax[j+4].set_ylim(-0.1, 1.1)
            ax[j+4].tick_params(axis='both', which='major', labelsize=12)

        elif j in [3, 8, 13, 18]:
            ax[j].pcolormesh(t, f, np.abs(Z_spec[jid]), 
                shading='gouraud', cmap='seismic', vmin=0, vmax=1
            )
            ax[j].set_xlim(0, 60)
            ax[j].yaxis.tick_right()

        for k in range(22):
            #ax[k].ticklabel_format(useMathText=False, axis='y', scilimits=(0,1))
            ax[k].yaxis.get_offset_text().set_fontsize(8)
            ax[k].tick_params(axis='both', which='major', labelsize=12, 
                direction='inout')
            ax[k].set_xlim(0, x.max())
            ax[k].set_ylabel(ylbl[k], fontsize=12)
            if k != 21:
                ax[k].set_xticks([])
    ax[20].annotate('local rockfall occurrence', (13, 0.2))
    ax[21].annotate('local earthquake occurrence', (13, 0.2))
    ax[20].plot(x, rfocc, linewidth=1.5)
    ax[20].set_ylim(-0.1, 1.1)
    ax[21].set_ylim(-0.1, 1.1)
    ax[21].plot(x, eqocc, linewidth=1.5)
    ax[21].set_xlabel("Time (s)")

    for ii in range(22):
        ax[ii].set_ylabel(ylbl[ii])
        if ii < 21:
            ax[ii].set_xlabel('')
            ax[ii].set_xticklabels('')
            
    ax[3].set_ylabel('Freq.\n(Hz)\n', fontsize=12)
    ax[19].legend(ncol=5, bbox_to_anchor=(1, 25), handletextpad=0.1,
        frameon=False, columnspacing=0.5)
    trans = ax[0].get_xaxis_transform()
    ax[-1].set_xlabel('Time (s)')
    plt.savefig(os.path.join(
        outfig, f"{str(stt)[:22]}.png"
    ))
    plt.close()
    #plt.show()