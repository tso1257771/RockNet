This readme file was generated on [2021-11-21] by [Wuyu Liao]

GENERAL INFORMATION

Title of Dataset: Seismic waveform collection of rockfall, earthquake, car-induced, and engineering signal in the Luhu tribe, Miaoli, Taiwan 

Author/Principal Investigator Information
Name: Wu-Yu Liao
ORCID: 0000-0003-2353-7006
Institution: National Cheng Kung University, Tainan City, TAIWAN
Address: No.1, Daxue Rd., East Dist., Tainan City 701, Taiwan(R.O.C.)Department of Earth Sciences
Email: tso1257771@gmail.com

Author/Associate or Co-investigator Information
Name: En-Jui Lee
ORCID: 0000-0003-1545-1640
Institution: Department of Earth Sciences, National Cheng Kung University, Tainan, Taiwan
Address: No.1, Daxue Rd., East Dist., Tainan City 701, Taiwan(R.O.C.)Department of Earth Sciences
Email: rickli92@gmail.com

Date of data collection: 2019-02-25 to 2020-08-26

Geographic location of data collection: Luhu tribe, Miaoli county, Taiwan

Information about funding sources that supported the collection of the data: The Ministry of Science and Technology, R.O.C., under Contract MOST 110-2116-M-006-011, 111-2116-M-006 -017, and MOST 109-2116-M-006-018. Partly supported by the Soil and Water Conservation Bureau, Taiwan, under Grant SWCB-108-294 and SWCB-109-227.


##### SHARING/ACCESS INFORMATION

Recommended citation for this dataset: 


##### DATA & FILE OVERVIEW

File List: <list all files (or folders, as appropriate for dataset organization) contained in the dataset, with a brief description>
.
├── benchmark				--> Scripts to perform benchmark test and print the results to screen
│   ├── P01_SingleSta_fusion_pred.py    --> Benchmark test of the single-detection model with both waveform and spectrogram as input
│   ├── P01_SingleSta_wf_pred.py  	--> Benchmark test of the single-detection model with waveform as input
│   ├── P02_Association_pred.py		--> Benchmark test of the association model with waveform as input
│   ├── P03_ConfuseMtx_local_single.py	--> Compute the benchmark results of the single-station detection model and print the results to screen
│   ├── P04_ConfuseMtx_local_network.py --> Compute benchmark results of the association model and print the results to screen
│   └── pred_results			--> Pre-computed prediction results of the single-station detection model and the association model.
├── data				--> Data storage and scripts to process data
│   ├── gen_TFRecord			--> Scripts to generate TFrecord files for TensorFlow throughput
│   ├── labeled_sac			--> The manual labeled SAC files
│   ├── Luhu_hdf5			--> The compiled HDF5 file and the scripts to reproduce ./labeled_sac
│   ├── metadata			--> Metadata of manual labels and data partition for model training, validation, and testing
│   ├── README.txt			--> Detailed architecture of this directory 
│   ├── tfrecord_Association		--> Generated TFRecord files for the association model using scripts in ./gen_TFRecord
│   └── tfrecord_SingleSta		--> Generated TFRecord files for the single-station detection model using scripts in ./gen_TFRecord
├── Luhu_pred_ex			--> A template of applying RockNet on hourly continuous data
│   ├── fig				--> Products of P02_plot.py
│   ├── net_pred			--> Outputs of P01_net_STMF.py, which is the prediction results of RockNet (formatted in SAC)
│   ├── P01_net_STMF.py			--> A prediction-making example of 
│   ├── P02_plot.py			--> An example of plotting the detection results
│   └── sac				--> The provided hourly SAC files
├── DATA_README.txt			--> This file
├── requirements.txt			--> The python package information for running the scripts
├── install_guide.txt			--> This python environment setting guide
├── tools				--> Scripts of the RockNet project
│   ├── build_model			--> Scripts of model building based on TensorFlow
│   ├── data_aug.py			--> Scripts of data augmentation
│   ├── data_utils.py			--> Scripts of data manipulations
│   ├── EQ_picker.py			--> Scripts of data manipulations
│   ├── mosaic_aug.py			--> Scripts of data augmentation
│   ├── network_RF_example_parser.py	--> Scripts of reading TFRecord for the association model
│   ├── RF_example_parser.py		--> Scripts of reading TFRecord for the single-station detection model
│   ├── rockfall_net_STMF_fusion.py	--> Scripts of handling continuous data for the association model
│   ├── rockfall_STMF_fusion.py		--> Scripts of handling continuous data for the single-station detection model
│   └── rockfall_utils.py		--> Scripts of data manipulations
└── trained_model			--> The trained models
    ├── fusion_RF_60s			--> The single-station detection model that feed waveform and spectrogram
    ├── net_fusion_RF_60s		--> The association model trained with additional dataset
    ├── net_fusion_RF_60s_noINSTANCE	--> The association model trained without additional dataset
    └── wf_RF_60s			--> The single-station detection model that feed waveform only


##### Relationship between files: 

The inputs of the scripts in `./benchmark` directory need the outputs of the scripts in `./data` directory, which are already prepared.


##### METHODOLOGICAL INFORMATION

### Description of methods used for collection/generation of data: <include links or references to publications or other documentation containing experimental design or protocols used in data collection>

The raw seismic waveforms were recorded by the Geophones and DATA-CUBE (https://digos.eu/wp-content/uploads/2020/11/2020-10-21-Broschure.pdf), and converted to `mseed` format with `cub2mseed` command (https://digos.eu/CUBE/DATA-CUBE-Download-Data-2017-06.pdf) of the CubeTools utility package (https://digos.eu/seismology/). 

The SAC software (Seismic Analysis Code, http://ds.iris.edu/ds/nodes/dmc/software/downloads/sac/102-0/) is used to process and visualize SAC files. 

The ObsPy (https://docs.obspy.org/) package is used to process and manipulate SAC files in python interface. 

The h5py package (https://docs.h5py.org/en/stable/) is used to storing seismic data and header information (i.e., metadata, including station and labeled information) in HDF5 (https://hdfgroup.org/) format for broader usages. 

The ObsPy and TensorFlow package (https://www.tensorflow.org/) are collaboratively used to convert the SAC files into the `TFRecord` format (https://www.tensorflow.org/tutorials/load_data/tfrecord) for TensorFlow applications.  

### Methods for processing the data: <describe how the submitted data were generated from the raw or collected data>

The `mseed` files into `SAC` format using the SAC software, and we provide only the minutes-long raw SAC files of different seismic sources (rockfall, earthquake, engineering, and car-induced signal). 

The PLOTPK command (http://www.adc1.iris.edu/files/sac-manual/commands/plotpk.html) of the SAC software is used to manual label the target waveform in the SAC file. 

The labeled information of each target waveform can be recognized as 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't0' in the header information of SAC file. For rockfall and car-induced waveforms, we denote ['t1', 't2], ['t3', 't4'], ..., ['t9', t0] to wrap the range of target signal in relative large signal-to-noise ratio. For earthquake waveforms, we labeled the seismic P and S phase arrivals with 't1' and 't2', separately. For engineering waveforms, no labels were made, in which the target signals dominate the visible part of the sliced SAC files. 

The SAC files and TFrecord files for performing benchmark test of our publication, RockNet: Rockfall and earthquake detection and association via multitask learning and transfer learning,  are available in the directories `./data/labeled_sac`, `./data/tfrecord_Association`, and `./data/tfrecord_SingleSta`.

However, one can reproduce the SAC files and TFrecord files from the provided codes:

1. To generate SAC files from the compiled HDF5 file (`./data/Luhu_hdf5/Luhu_dataset.h5`), you need enter the directory `./data/Luhu_hdf5`, and execute the script `./data/Luhu_hdf5/h52sac.py`. The generated SAC files would be available in `./data/labeled_sac`. 

> cd ./data/Luhu_hdf5 

> python h52sac.py

2. For benchmark dataset reproduction from the SAC files in `./data/labeled_sac`, you need to enter `./data/gen_TFrecord`, where we place the scripts to generate TFRecord files for building the RockNet.  The TFrecord files are/would be availabe in './data/tfrecord_Association/', './data/tfrecord_SingleSta/'.

> cd ./data/gen_TFrecord

# generate rockfall waveform benchmarking the single-station detection model 
# Outputs: './data/tfrecord_SingleSta/RF_multiple_sta_label'
> python P01.3_RF_lbl_across_stas_test.py

# generate earthquake waveform for benchmarking the single-station detection model 
# Outputs: './data/tfrecord_SingleSta/EQ'
> python P03.3_eq_lbl_across_stas_test.py

# generate car-induced waveform for benchmarking the single-station detection model 
# Outputs: './data/tfrecord_SingleSta/car_multiple_sta_label'
> python P04.3_car_lbl_across_stas_test.py

# generate engineering waveform for benchmarking the single-station detection model
# Outputs: './data/tfrecord_SingleSta/engineering'
> python P05.3_engineering_test.py

# generate rockfall waveform benchmarking the association model
# Outputs: './data/tfrecord_Association/RF'
> python P06.3_network_RF_test.py

# generate earthquake waveform benchmarking the association model
# Outputs: './data/tfrecord_Association/EQ'
> python P07.3_network_EQ_lbl_test.py

Noted that the generation of engineering and car-induced signal waveform for the association model needs the hour-long SAC files, which are not available due to the large volume. We only provide the TFrecord in our benchmark test: './data/tfrecord_Association/car', './data/tfrecord_SingleSta/engineering'.


##### Instrument- or software-specific information needed to interpret the data: <include full name and version of software, and any necessary packages or libraries needed to run scripts>

All the scripts are build with python (3.7.3) under Linux. We recommend Anaconda (https://www.anaconda.com/products/distribution/) and pip (https://pypi.org/project/pip/) for simple package management and environment setting.

obspy==1.2.2
h5py==3.1.0
numpy==1.19.5
scipy==1.5.2
pandas==1.3.5
tensorflow==2.5.3
tensorflow_addons==0.11.2
matplotlib==3.3.2

Standards and calibration information, if appropriate: 


Environmental/experimental conditions: 


Describe any quality-assurance procedures performed on the data: We have manually checked every sample


People involved with sample collection, processing, analysis and/or submission: Wu-Yu Liao and En-Jui Lee



