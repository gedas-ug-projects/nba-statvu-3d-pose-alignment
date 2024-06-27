#!/usr/bin/env bash
set -e
# remeber to do this for cuda export CUDA_HOME=/usr/local/cuda ,export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

#export CONDA_ENV_NAME=slahmr
export CONDA_ENV_NAME=pose_alignment_slahmr

# Ensure conda is available in the current shell session
source $(conda info --base)/etc/profile.d/conda.sh

# Create the conda environment with Python 3.10
conda create -n $CONDA_ENV_NAME python=3.10 -y

conda activate $CONDA_ENV_NAME

#Set CUDA_HOME env var 
export CUDA_HOME=/usr/local/cuda-12.4

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH 

export PATH=$CUDA_HOME/bin:$PATH
conda install -c conda-forge cudatoolkit=11.7 cudnn=8.1.0 -y

# Install PyTorch using conda with appropriate CUDA drivers
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# install pytorch using pip, update with appropriate cuda drivers if necessary
#original: pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu117
# uncomment if pip installation isn't working
# conda install pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# install pytorch scatter using pip, update with appropriate cuda drivers if necessary
# original: pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
# uncomment if pip installation isn't working
# conda install pytorch-scatter -c pyg -y

# install PHALP
pip install phalp[all]@git+https://github.com/brjathu/PHALP.git

# install remaining requirements
pip install -r requirements.txt

# install source
pip install -e .

# install ViTPose
pip install -v -e third-party/ViTPose

# install DROID-SLAM
cd third-party/DROID-SLAM
python setup.py install
cd ../..
