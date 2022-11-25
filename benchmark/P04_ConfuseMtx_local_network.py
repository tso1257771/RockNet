import os
import logging
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO,                
    format='%(levelname)s : %(asctime)s : %(message)s')

detect_thre = 0.5
#'net_fusion_RF_60s' 'net_fusion_RF_60s_noINSTANCE'
mdl_hdr = 'net_fusion_RF_60s' 
print(f"\n{mdl_hdr}")
bench_dir = os.path.join('./pred_results', mdl_hdr)
# rockfall dataset
RF_df_dir = os.path.join(
    bench_dir, 'Association_pred_rockfall.txt')
eq_df_dir = os.path.join(
    bench_dir, 'Association_pred_earthquake.txt')
car_df_dir = os.path.join(
    bench_dir, 'Association_pred_nz_car.txt')
engineering_df_dir = os.path.join(
    bench_dir, 'Association_pred_nz_engineering.txt')

#net_fusion_RF_60s_noINSTANCE_scratch, net_fusion_RF_60s
logging.info(f"\n----------\n{mdl_hdr}\n----------")
rf_df = pd.read_table(RF_df_dir,
    names=['evid', 'label_type', 'pred_type', 'ev_occ_pt', 'pred_occ_pt',
    'eqocc_prob', 'rfocc_prob'], delimiter=',', header=0)

eq_df = pd.read_table(eq_df_dir,
    names=['evid', 'label_type', 'pred_type', 'ev_occ_pt', 'pred_occ_pt',
    'eqocc_prob', 'rfocc_prob'], delimiter=',', header=0)

car_df = pd.read_table(car_df_dir,
    names=['evid', 'label_type', 'pred_type', 'ev_occ_pt', 'pred_occ_pt',
    'eqocc_prob', 'rfocc_prob'], delimiter=',', header=0)
engineering_df = pd.read_table(engineering_df_dir,
    names=['evid', 'label_type', 'pred_type', 'ev_occ_pt', 'pred_occ_pt',
    'eqocc_prob', 'rfocc_prob'], delimiter=',', header=0)
nz_df = pd.concat([car_df, engineering_df])

## componentes of confusion matrix
#                   Ground truth 
#------------------------------------------------
#               |earthquake  |rockfall    |others
#------------------------------------------------
# earthquake    |a           |b           |c
#------------------------------------------------
# rockfall      |d           |e           |f
#------------------------------------------------
# others        |g           |h           |i
#
a = len(eq_df[
    np.logical_and(
        eq_df['pred_type'].str.strip()=='earthquake',
        eq_df['eqocc_prob'] >= detect_thre
    )
])
b = len(rf_df[
    np.logical_and(
        rf_df['pred_type'].str.strip()=='earthquake',
        rf_df['eqocc_prob'] >= detect_thre
    )
])
c = len(nz_df[
    np.logical_and(
        nz_df['pred_type'].str.strip()=='earthquake',
        nz_df['eqocc_prob'] >= detect_thre
    )
])
d = len(eq_df[
    np.logical_and(
        eq_df['pred_type'].str.strip()=='rockfall',
        eq_df['rfocc_prob'] >= detect_thre
    )
])
e = len(rf_df[
    np.logical_and(
        rf_df['pred_type'].str.strip()=='rockfall',
        rf_df['rfocc_prob'] >= detect_thre
    )
])
f = len(nz_df[
    np.logical_and(
        nz_df['pred_type'].str.strip()=='rockfall',
        nz_df['rfocc_prob'] >= detect_thre
    )
])
g = len(eq_df[eq_df['pred_type'].str.strip()=='noise']) + \
    len(eq_df[
        np.logical_and(
            eq_df['pred_type'].str.strip()=='earthquake',
            eq_df['eqocc_prob'] < detect_thre
        )
    ])+\
    len(eq_df[
        np.logical_and(
            eq_df['pred_type'].str.strip()=='rockfall',
            eq_df['rfocc_prob'] < detect_thre
        )
    ])
h = len(rf_df[rf_df['pred_type'].str.strip()=='noise']) +\
    len(rf_df[
        np.logical_and(
            rf_df['pred_type'].str.strip()=='rockfall',
            rf_df['rfocc_prob'] < detect_thre
        )
    ]) +\
    len(rf_df[
            np.logical_and(
                rf_df['pred_type'].str.strip()=='earthquake',
                rf_df['eqocc_prob'] < detect_thre
            )
    ])
i = len(nz_df[nz_df['pred_type'].str.strip()=='noise'])+\
    len(nz_df[
        np.logical_and(
            nz_df['pred_type'].str.strip()=='rockfall',
            nz_df['rfocc_prob'] < detect_thre
        )
    ])+\
    len(nz_df[
    np.logical_and(
        nz_df['pred_type'].str.strip()=='earthquake',
        nz_df['eqocc_prob'] < detect_thre
    )
    ])

assert len(nz_df) == c+f+i
assert len(rf_df) == b+e+h
assert len(eq_df) == a+d+g

# class EQ
eq_TP = a
eq_TN = e+f+h+i 
eq_FP = b+c
eq_FN = d+g

# class rf
rf_TP = e
rf_TN = a+c+g+i
rf_FP = d+f
rf_FN = b+h
# class nz
nz_TP = i
nz_TN = a+b+d+e
nz_FP = g+h
nz_FN = c+f

# confusion matrix
eq_precision = eq_TP / (eq_TP + eq_FP)
eq_recall = eq_TP / (eq_TP + eq_FN)
eq_F1 = 2*eq_precision*eq_recall / (eq_precision+eq_recall)
logging.info(f"@ earthquakes samples: {len(eq_df)}")
logging.info(f"earthquake precision : {eq_precision:.4f}")
logging.info(f"earthquake recall : {eq_recall:.4f}")
logging.info(f"earthquake F1-score : {eq_F1:.4f}")
logging.info("--------------------")

rf_precision = rf_TP / (rf_TP + rf_FP)
rf_recall = rf_TP / (rf_TP + rf_FN)
rf_F1 = 2*rf_precision*rf_recall / (rf_precision+rf_recall)
logging.info(f"@rockfall samples: {len(rf_df)}")
logging.info(f"rockfall precision : {rf_precision:.4f}")
logging.info(f"rockfall recall : {rf_recall:.4f}")
logging.info(f"rockfall F1-score : {rf_F1:.4f}")
logging.info("--------------------")

nz_precision = nz_TP / (nz_TP + nz_FP)
nz_recall = nz_TP / (nz_TP + nz_FN)
nz_F1 = 2*nz_precision*nz_recall / (nz_precision+nz_recall)
logging.info(f"@ noise samples: {len(nz_df)}")
logging.info(f"noise precision : {nz_precision:.4f}")
logging.info(f"noise recall : {nz_recall:.4f}")
logging.info(f"noise F1-score : {nz_F1:.4f}")
logging.info("--------------------")


micro_TP = eq_TP + rf_TP + nz_TP
micro_FP = eq_FP + rf_FP + nz_FP
micro_FN = eq_FN + rf_FN + nz_FN
micro_recall = micro_TP / (micro_TP+micro_FN)
micro_precision = micro_TP / (micro_TP+micro_FP)

micro_F1 = 2*micro_recall*micro_precision/(micro_recall+micro_precision)
macro_F1 = (eq_F1 + rf_F1 + nz_F1)/3
weighted_F1 = (eq_F1*len(eq_df) + rf_F1*len(rf_df) +\
    nz_F1*len(nz_df))/(len(eq_df)+len(rf_df)+len(nz_df))
logging.info(f"micro F1-score: {micro_F1:.4f}")
logging.info(f"macro F1-score: {macro_F1:.4f}")
logging.info(f"weighted F1-score: {weighted_F1:.4f}")
