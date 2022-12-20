
##### DATA & FILE OVERVIEW

# Please download the archived data `RockNet_data.zip` from the following DOI:

https://doi.org/10.5061/dryad.tx95x6b2f

# and decompress the downloaded file in the directory under `RockNet/data` 

$ mv DOWNLOAD_PATH/doi_10.5061_dryad.tx95x6b2f*.zip DOWNLOAD_PATH/README.md ./data

$ cd ./data

$ unzip doi_10.5061_dryad.tx95x6b2f*.zip

$ unzip RockNet_data.zip

# the decompressed directories and files are: `labeled_sac`, `tfrecord_Association`, `tfrecord_SingleSta`, and `Luhu_dataset.h5`;
# move the .h5 file to the directory `./data/Luhu_hdf5`

$ mv Luhu_dataset.h5 ./data/Luhu_hdf5

# File List: 
.
├── benchmark				--> Scripts to perform benchmark test and print the results to screen
│   ├── P01_SingleSta_fusion_pred.py    --> Benchmark test of the single-detection model with both waveform and spectrogram as input
│   ├── P01_SingleSta_wf_pred.py  	--> Benchmark test of the single-detection model with waveform as input
│   ├── P02_Association_pred.py		--> Benchmark test of the association model with waveform as input
│   ├── P03_ConfuseMtx_local_single.py	--> Compute the benchmark results of the single-station detection model and print the results to screen
│   ├── P04_ConfuseMtx_local_network.py --> Compute benchmark results of the association model and print the results to screen
│   └── pred_results			--> Pre-computed prediction results of the single-station detection model and the association model.
├── data				--> Data storage and scripts to process data
│   ├── gen_TFRecord		--> Scripts to generate TFrecord files for TensorFlow throughput
│   ├── labeled_sac			--> The manual labeled SAC files (host on Dryad)
│   ├── Luhu_hdf5			--> The compiled HDF5 file and the scripts to reproduce ./labeled_sac (Luhu_dataset.h5 is host on Dryad)
│   ├── metadata			--> Metadata of manual labels and data partition for model training, validation, and testing
│   ├── tfrecord_Association		--> Generated TFRecord files for the association model using scripts in ./gen_TFRecord
│   └── tfrecord_SingleSta		--> Generated TFRecord files for the single-station detection model using scripts in ./gen_TFRecord
├── Luhu_pred_ex			--> A template of applying RockNet on hourly continuous data
│   ├── fig				--> Products of P02_plot.py
│   ├── net_pred			--> Outputs of P01_net_STMF.py, which is the prediction results of RockNet (formatted in SAC)
│   ├── P01_net_STMF.py			--> A prediction-making example of 
│   ├── P02_plot.py			--> An example of plotting the detection results
│   └── sac				--> The provided hourly SAC files
├── DIR_README.md			--> This file. Descriptions of directory components in detail.
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


