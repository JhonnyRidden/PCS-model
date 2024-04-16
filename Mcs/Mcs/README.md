# MCS

## 1. Introduction
This is the official implementation of Mcs. 

## 2. Setup
- Install CUDA 10.1
- run `setup_py.sh` to install the necessary dependecies for python environments.

## 3. Dataset
Due to the data policy of Google and OxCGRT, we cannot directly provide the original data. 
However, we provide a data pulling and preprocessing script. 
You can run the following command to download data from official website.
```shell
cd ~/Mcs/src
python data_utils.py
# Pulling epidemic data from Google, CSSE and OxCGRT. 
# The downloaded data can be found in ~/Mcs/data/
```


## 4. Usage
To run MCS, please follow the steps below:
```shell
cd ~/Mcs/src
python run_models.py --forecast_date 2021-05-01 
# Running ensemble for Mcs
```

