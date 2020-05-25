#!/bin/bash

# Variables
NAT_DIR='naturally_trained'
ADV_DIR='adv_trained'

# Download naturally-trained model
if [ ! -d "$NAT_DIR" ]; then
  echo '-----------------------------------'
  echo 'Downloading naturally-trained model'
  echo '-----------------------------------'
  wget 'https://www.dropbox.com/s/cgzd5odqoojvxzk/natural.zip'
  echo '---------------------------------'
  echo 'unzipping naturally-trained model'
  echo '---------------------------------'
  unzip natural.zip
  echo '--------------------------'
  echo 'Deleting unnecessary files'
  echo '--------------------------'
  mv ./models/naturally_trained ./
  rm -rf ./models
  rm natural.zip
else
  echo '--------------------------------------'
  echo 'Naturally-trained model already exists'
  echo '--------------------------------------'
fi

# Download adversarially-trained model
if [ ! -d "$ADV_DIR" ]; then
  echo '---------------------------------------'
  echo 'Downloading adversarially-trained model'
  echo '---------------------------------------'
  wget 'https://www.dropbox.com/s/g4b6ntrp8zrudbz/adv_trained.zip'
  echo '-------------------------------------'
  echo 'unzipping adversarially-trained model'
  echo '-------------------------------------'
  unzip adv_trained.zip
  echo '--------------------------'
  echo 'Deleting unnecessary files'
  echo '--------------------------'
  mv ./models/adv_trained ./
  rm -rf ./models
  rm adv_trained.zip
else
  echo '------------------------------------------'
  echo 'Adversarially-trained model already exists'
  echo '------------------------------------------'
fi

