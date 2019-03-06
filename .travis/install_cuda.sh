#!/bin/bash

set -ev

###########################
# Install Cuda
###########################

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_10.0.130-1_amd64.deb
dpkg -i cuda-repo-ubuntu1404_10.0.130-1_amd64.deb
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub
apt-get update -qq
apt-get install -qq build-essential libffi-dev pkg-config cuda-nvrtc-dev-10-0
mkdir -p /usr/local/cuda-10.0/include/
ln -s /usr/local/cuda-10.0/targets/x86_64-linux/include/nvrtc.h /usr/local/cuda-10.0/include/nvrtc.h
