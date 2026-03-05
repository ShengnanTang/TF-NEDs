This code is the official PyTorch implementation of our Paper: Frequency Guided Neural ODE: A Framework for Multivariate Time Series Forecasting.

If you find this project helpful, please don't forget to give it a ⭐ Star to show your support. Thank you!


## Quickstart

### Installation
```
pip install -r requirements.txt
```
### Data preparation
Prepare Data. You can obtained the well pre-processed datasets at https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto. 
Then place the downloaded data under the folder ```./data/```.
```
data
|   |-- PEMSD4
|   |   |-- PEMSD4.npz
```
### Train and evaluate model
1. The model structure of **TF-NDEs** under the folder ```./model/GCDE.py/```
2. We provide the scripts for TF-NDEs under the folder ```./model/```. For example you can reproduce a experiment result as the following:
```
cd model
python Run_cde.py
```

## Community Support | Acknowledgements

This project is built on the shoulders of the open-source community.  
Special thanks to the authors and contributors of the following repositorie:

-https://github.com/jeongwhanchoi/STG-NCDE
