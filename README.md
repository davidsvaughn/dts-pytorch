# dts-pytorch
Deep Text Scoring - PyTorch

pytorch setup
```
virtualenv --system-site-packages -p python3.6 ./venv-pytorch
cd venv-pytorch
source bin/activate

pip install -U numpy
pip install -U pandas
pip install -U sklearn
pip install -U scipy
pip install -U matplotlib

pip3 install https://download.pytorch.org/whl/cu80/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
```