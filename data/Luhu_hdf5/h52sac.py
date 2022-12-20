import os
import shutil
import logging
import h5py
import numpy as np
from glob import glob
from obspy import read
from obspy.io.sac.sactrace import SACTrace
logging.basicConfig(level=logging.INFO,
    format='%(levelname)s : %(asctime)s : %(message)s')

hf = h5py.File('./Luhu_dataset.h5', 'r')
outpath = '../labeled_sac'
if os.path.exists(outpath):
    shutil.rmtree(outpath)

dirs = list(hf)
for i in range(len(dirs)):
    evid = list(hf[dirs[i]])
    for j in range(len(evid)):
        logging.info(f"Processing class {i+1}/{len(dirs)}: {dirs[i]}")
        logging.info(f"Events {j+1}/{len(evid)}: {evid[j]}")

        outdir = os.path.join(outpath, dirs[i], evid[j])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        file_id = list(hf[dirs[i]][evid[j]])
        for k in range(len(file_id)):
            keys = list(hf[dirs[i]][evid[j]][file_id[k]])
            keys.remove('data')

            _data = np.array(
                hf[dirs[i]][evid[j]][file_id[k]]['data']).astype(float)
            _hdr_dict = dict()
            for l in range(len(keys)):
                _value = hf[dirs[i]][evid[j]][file_id[k]][keys[l]][()]
                if type(_value) == bytes:
                    _value = _value.decode()
                _hdr_dict[keys[l]] = _value
            _new_sac = SACTrace(data=_data, **_hdr_dict)
            outname = os.path.join(outdir, f"{file_id[k]}.sac")
            _new_sac.write(outname)
