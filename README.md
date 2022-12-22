[![DOI](https://zenodo.org/badge/565642324.svg)](https://zenodo.org/badge/latestdoi/565642324)
# RockNet
Rockfall and earthquake detection and association via multitask learning and transfer learning
Our preprint article can be found [here](https://essopenarchive.org/doi/full/10.22541/essoar.167160646.63337688/v1).

![2020-03-28T13:41:20 00](https://user-images.githubusercontent.com/30610646/203888301-ba149105-6701-43b7-a2fe-8c7be1852894.png)

## Complete dataset
Please also download the complete data hosted on Dryad (https://doi.org/10.5061/dryad.tx95x6b2f),
follow the instructions and place the files to specified directories in this repository.

## Summary

* [Installation](#installation)
* [Make prediction on hourly SAC files](#Make-prediction-on-hourly-SAC-files)

### Installation
To run this repository, we suggest Anaconda and pip for environment managements.

Clone this repository:

```bash
git clone https://github.com/tso1257771/RockNet.git
cd RockNet
```

Create a new environment 

```bash
conda create -n rocknet python==3.7.3 anaconda
conda activate rocknet
pip install --upgrade pip
pip install -r ./requirements.txt --ignore-installed
```

### Make prediction on hourly SAC files
In this repository, we provide two hourly three-component seismograms as examples for making predictions on continuous data.<br />
The data seismograms were collected in the Luhu tribe, Miaoli county, Taiwan.<br />

Enter the directory  ```./Luhu_pred_ex```<br />
```bash
cd ./Luhu_pred_ex
```
1. Run script ```Luhu_pred_ex/P01_net_STMF.py``` to generate the output functions (also in SAC format) in ```Luhu_pred_ex/net_pred``` from the provided SAC files ```Luhu_pred_ex/sac```<br />
```
python P01_net_STMF.py
```
2. Plot some prediction results<br />
```
python P02_plot.py
```





