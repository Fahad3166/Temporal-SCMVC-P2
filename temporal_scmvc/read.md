# P1 SCMVC / Temporal SCMVC Run Guide

This repository contains the original SCMVC implementation and the temporal extensions used for the P1 project:

- Original SCMVC baseline with fully connected per-view encoders.
- SCMVC-TCN with temporal convolution encoders for multi-view time-series data.
- SCMVC-LSTM with recurrent temporal encoders for multi-view time-series data.

The temporal experiments were tested on HAR, MHEALTH, PAMAP2, and WISDM.

## 1. Environment Setup

From the SCMVC root folder:

```bash
conda create -n scmvc python=3.9
conda activate scmvc

pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn tqdm matplotlib

## requirment 
python==3.7.13
pytorch==1.12.0
numpy==1.21.5
scikit-learn==0.22.2.post1
scipy==1.7.3
```



## 2. Run Original SCMVC

The original SCMVC code is in the root folder. It uses the original flattened multi-view inputs.

Example commands:

```bash
python train.py 


The original datasets are loaded from:

```text
data/
```

## 3. Run Temporal SCMVC-TCN

The TCN version is in:

temporal_scmvc/train_temporal_tcn.py


Move into the temporal folder:


cd temporal_scmvc


General command format:

```

python train_temporal_tcn.py --dataset DATASET_NAME --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50

```

Supported dataset names:



har
mhealth
pamap2
wisdm


For WISDM, I used only 20 files per view to keep the experiment manageable on MacBook Air:

```bash
python train_temporal_tcn.py --dataset wisdm --wisdm_max_files_per_view 20 --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50
```

## 4. Run Temporal SCMVC-LSTM

The LSTM version is in:


temporal_scmvc/train_temporal_lstm.py



Move into the temporal folder:


cd temporal_scmvc


General command format:

```bash
python train_temporal_lstm.py --dataset DATASET_NAME --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50
```

For WISDM:

```bash
python train_temporal_lstm.py --dataset wisdm --wisdm_max_files_per_view 20 --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50
```

## 5. Final Commands 

Run these commands from:

```bash
cd temporal_scmvc
```

### WISDM

TCN:

```bash
python train_temporal_tcn.py --dataset wisdm --wisdm_max_files_per_view 20 --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50
```

LSTM:

```bash
python train_temporal_lstm.py --dataset wisdm --wisdm_max_files_per_view 20 --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50
```

### PAMAP2

TCN:

```bash
python train_temporal_tcn.py --dataset pamap2 --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50
```

LSTM:

```bash
python train_temporal_lstm.py --dataset pamap2 --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50
```

### MHEALTH

TCN:

```bash
python train_temporal_tcn.py --dataset mhealth --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50
```

LSTM:

```bash
python train_temporal_lstm.py --dataset mhealth --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50
```

### HAR

TCN:

```bash
python train_temporal_tcn.py --dataset har --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50
```

LSTM:

```bash
python train_temporal_lstm.py --dataset har --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50
```

## 6. Local Window-Level Contrastive Loss

This was tested only with the TCN model.

WISDM local contrastive loss:

```bash
python train_temporal_tcn.py --dataset wisdm --wisdm_max_files_per_view 20 --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50 --local_loss_weight 0.1 --local_chunk_size 50
```

PAMAP2 local contrastive loss:

```bash
python train_temporal_tcn.py --dataset pamap2 --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50 --local_loss_weight 0.1 --local_chunk_size 10
```

## 7. Dataset Paths Expected by the Code

The temporal loaders expect the datasets in these locations:

```text
HAR:
HAR_dataset/UCI HAR Dataset/

MHEALTH:
data/MHEALTHDATASET/

PAMAP2:
data/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol/

WISDM:
data/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/
```


## 8. Main Results From Final Experiments

| Dataset | Best Model | ACC | NMI | PUR |
|---|---:|---:|---:|---:|
| WISDM | TCN | 0.4154 | 0.4977 | 0.4316 |
| PAMAP2 | TCN | 0.4596 | 0.4144 | 0.4714 |
| MHEALTH | LSTM | 0.5776 | 0.5993 | 0.6062 |
| HAR | LSTM | 0.6758 | 0.5740 | 0.6884 |

Summary:

- TCN performed better on WISDM and PAMAP2, where local short-window motion patterns are important.
- LSTM performed better on MHEALTH and HAR, where longer temporal dependencies and sequential activity transitions were more useful.
- Local window-level contrastive loss was implemented and tested, but it did not improve the final TCN results compared with the standard global contrastive objective.
