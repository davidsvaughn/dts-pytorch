# dts-pytorch
Deep Text Scoring - PyTorch

## PyTorch Setup
```
virtualenv -p python3 venv
source venv/bin/activate

pip install -U numpy
pip install -U pandas
pip install -U sklearn
pip install -U scipy
pip install -U matplotlib

pip3 install torch torchvision
```

## Run
```
cd python
python -u train.py | tee log.txt
```