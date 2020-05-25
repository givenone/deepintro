#!/bin/bash

# Variables
DATA_DIR='cifar-10-batches-py'

# Download Cifar-10 dataset
if [ ! -d "$DATA_DIR" ]; then
  echo '----------------------------'
  echo 'Downloading Cifar-10 dataset'
  echo '----------------------------'
  wget 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  echo '----------------------------'
  echo 'Unzipping Cifar-10 dataset'
  echo '----------------------------'
  tar -xvzf cifar-10-python.tar.gz
  echo '----------------------------'
  echo 'Deleting unnecessary files'
  echo '----------------------------' 
  rm -rf cifar-10-python.tar.gz
else
  echo '------------------------------'
  echo 'Cifar-10 dataset already exits'
  echo '------------------------------'
fi
