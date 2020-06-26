#!/bin/bash
# GFO 04/06/2020

# get and install miniconda latest version
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p $HOME/miniconda3
bash ./$HOME/miniconda3/conda init

# create env and install what's necessary
bash conda update -n base -c defaults conda
bash conda create -n gymenv python=3.6 pip
bash conda activate gymenv
bash sudo apt-get update && sudo apt-get install libopenmpi-dev
bash git clone https://github.com/openai/spinningup.git
bash pip install -e spinningup/.
bash pip install gym
bash pip install -e ~/ML/gym-ShipNavigation/.
bash pip install -e ~/ML/gym-ShipNavigation/.


bash conda create -n torchenv python=3.6 pip
bash conda activate torchenv
bash conda install pytorch torchvision cpuonly -c pytorch
