# dts-pytorch
Deep Text Scoring - in PyTorch

- this code implements a _flat_ (not hierarchical) RNN model on _character_ sequences
- so far, only binary classification has been implemented (for insufficient filter)
- use dts-tf as a guide to extend to qwk regression

## Get data from spshare
copy files from: //spshare/users/dvaughn/dts-pytorch/data -> ./data

## PyTorch Setup
```
virtualenv -p python3 venv
source venv/bin/activate

pip install -U numpy
pip install -U pandas
pip install -U sklearn
pip install -U scipy
pip install -U matplotlib
pip install -U pprint

pip3 install torch torchvision
```

## Set Config Parameters:
- line 33: `group = 'fw' # rs=research | bw= brief write | fw = full write`
- lines 180-200: all other parameters (DNN model, training, etc)

## Run
```
cd python
python -u train.py | tee log.txt
```
