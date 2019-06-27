# dts-pytorch
Deep Text Scoring - in PyTorch

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

## Run
```
cd python
python -u train.py | tee log.txt
```