#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:19:26 2018

@author: s164435
"""
import tensorflow as tf
from models_cyclegan_patchD import CycleGAN
from config_cyclegan_patchD import get_config
from ops_cyclegan_patchD import mkdir
import os
import glob
import numpy as np
import time

def fetch_training_data_dirs(data_dir):
    training_data_dirs = list()
    for subject in sorted(glob.glob(os.path.join(data_dir, "*"))):
        training_data_dirs.append(subject)
    return training_data_dirs

def main():
  sess = tf.Session()
  config = get_config(is_train=True) #read config file
  mkdir(config.tmp_dir)
  mkdir(config.ckpt_dir)

  #build CycleGAN model
  reg = CycleGAN(sess, config, "CycleGAN", is_train=True)

  #Get training and validation data directories
  prostate_train_cbct_dirs=fetch_training_data_dirs(config.prostate_data_train_cbct)
  prostate_train_ct_dirs=fetch_training_data_dirs(config.prostate_data_train_ct)
  prostate_validate_cbct_dirs=fetch_training_data_dirs(config.prostate_data_validate_cbct)
  prostate_validate_ct_dirs=fetch_training_data_dirs(config.prostate_data_validate_ct)  
  
  hn_train_cbct_dirs=fetch_training_data_dirs(config.hn_data_train_cbct)
  hn_train_ct_dirs=fetch_training_data_dirs(config.hn_data_train_ct)
  hn_validate_cbct_dirs=fetch_training_data_dirs(config.hn_data_validate_cbct)
  hn_validate_ct_dirs=fetch_training_data_dirs(config.hn_data_validate_ct)          

  train_cbct_dirs=prostate_train_cbct_dirs+hn_train_cbct_dirs
  train_ct_dirs=prostate_train_ct_dirs+hn_train_ct_dirs
  validate_cbct_dirs=prostate_validate_cbct_dirs+hn_validate_cbct_dirs
  validate_ct_dirs=prostate_validate_ct_dirs+hn_validate_ct_dirs
  
  train_ct_index=np.arange(len(train_ct_dirs))
  train_cbct_index=np.arange(len(train_cbct_dirs))  
  validate_ct_index=np.arange(len(validate_ct_dirs))
  validate_cbct_index=np.arange(len(validate_cbct_dirs))

  batch_ct=np.float32(np.zeros((config.batch_size,config.im_size[0],config.im_size[1],config.channel)))
  batch_cbct=np.float32(np.zeros((config.batch_size,config.im_size[0],config.im_size[1],config.channel)))
  batch_ct_validation=np.float32(np.zeros((config.batch_size,config.im_size[0],config.im_size[1],config.channel)))
  batch_cbct_validation=np.float32(np.zeros((config.batch_size,config.im_size[0],config.im_size[1],config.channel)))
  
  counter = 1
  start_time = time.time()
   
  #Start to train CycleGAN model
  for epoch in range(config.epoch):
      # random shuffle images for each epoch
      np.random.shuffle(train_ct_index)
      np.random.shuffle(train_cbct_index)

      batch_idxs = min(train_ct_index.shape[0], train_cbct_index.shape[0]) // config.batch_size
      lr = config.lr if epoch < config.epoch_step else config.lr*(config.epoch-epoch)/(config.epoch-config.epoch_step)
       
      for idx in range(0, batch_idxs):      
          batch_cbct_index = train_cbct_index[idx*config.batch_size:(idx + 1)*config.batch_size]
          batch_ct_index=train_ct_index[idx*config.batch_size:(idx + 1)*config.batch_size]
          random_choice_cbct=np.random.choice(validate_cbct_index, config.batch_size,replace=False)
          random_choice_ct=np.random.choice(validate_ct_index, config.batch_size,replace=False)
          
          for batchidx in range(config.batch_size):
              #read ct image in HU unit from raw data file for training
              fname=train_ct_dirs[batch_ct_index[batchidx]]
              with open(fname, 'r') as infile:
                  infile.seek(0,0)
                  ct = np.fromfile(infile, dtype='f')
              batch_ct[batchidx,:,:,0]=ct.reshape(512,512)

              # read cbct image in HU unit from raw data file for training
              fname=train_cbct_dirs[batch_cbct_index[batchidx]]
              with open(fname, 'r') as infile:
                  infile.seek(0,0)
                  cbct = np.fromfile(infile, dtype='f')
              batch_cbct[batchidx,:,:,0]=cbct.reshape(512,512)

              # read ct image in HU unit from raw data file for validation
              fname=validate_ct_dirs[random_choice_ct[batchidx]]
              with open(fname, 'r') as infile:
                  infile.seek(0,0)
                  ct = np.fromfile(infile, dtype='f')
              batch_ct_validation[batchidx,:,:,0]=ct.reshape(512,512)

              # read cbct image in HU unit from raw data file for validation
              fname=validate_cbct_dirs[random_choice_cbct[batchidx]]
              with open(fname, 'r') as infile:
                  infile.seek(0,0)
                  cbct = np.fromfile(infile, dtype='f')
              batch_cbct_validation[batchidx,:,:,0]=cbct.reshape(512,512)
              
              # data normalization
              batch_cbct=np.where(batch_cbct>=1500,(batch_cbct-1500)*(3071-1500)/5500+1500,batch_cbct)
              batch_cbct=np.tanh(batch_cbct/1000)
              batch_cbct_validation=np.where(batch_cbct_validation>=1500,(batch_cbct_validation-1500)*(3071-1500)/5500+1500,batch_cbct_validation)
              batch_cbct_validation=np.tanh(batch_cbct_validation/1000)
              batch_ct=np.tanh(batch_ct/1000) 
              batch_ct_validation=np.tanh(batch_ct_validation/1000)

          
          reg.fit(batch_cbct, batch_ct,lr,counter,batch_cbct_validation,batch_ct_validation)
          counter += 1
          print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch_idxs, time.time() - start_time)))
          if counter % 20 == 0:
              reg.test(config.tmp_dir, batch_cbct, batch_ct)
              reg.save(config.ckpt_dir)



if __name__ == "__main__":
  main()

