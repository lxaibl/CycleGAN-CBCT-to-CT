#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:11:47 2018

@author: s164435
"""

class Config(object):
  pass

def get_config(is_train):
  config = Config()
  if is_train:
    config.batch_size = 1
    config.im_size = [512, 512]
    config.channel=1
    config.lr = 0.0002
    config.epoch=200
    config.epoch_step=100
    config.weight=10.0
    config.weight_identity=5.0
    config.continue_train= False
    config.pool_size = 50
    
    config.tmp_dir = "/home/xiao/CycleGAN/tmp_cyclegan_patchD"
    config.ckpt_dir = "/mnt/md0/xiao/ckpt_cyclegan_patchD"
    
    config.hn_data_train_cbct="/mnt/md0/xiao/data_cyclegan_patchD/train_HN/CBCT/"
    config.hn_data_train_ct="/mnt/md0/xiao/data_cyclegan_patchD/train_HN/CT/"
    config.hn_data_validate_cbct="/mnt/md0/xiao/data_cyclegan_patchD/validate_HN/CBCT/"
    config.hn_data_validate_ct="/mnt/md0/xiao/data_cyclegan_patchD/validate_HN/CT/"
    
    config.prostate_data_train_cbct="/mnt/md0/xiao/data_cyclegan_patchD/train_prostate/CBCT/"
    config.prostate_data_train_ct="/mnt/md0/xiao/data_cyclegan_patchD/train_prostate/CT/"
    config.prostate_data_validate_cbct="/mnt/md0/xiao/data_cyclegan_patchD/validate_prostate/CBCT/"
    config.prostate_data_validate_ct="/mnt/md0/xiao/data_cyclegan_patchD/validate_prostate/CT/"
    
    config.logs = "/mnt/md0/xiao/logs_cyclegan_patchD"
    
  else:
    config.batch_size = 1
    config.im_size = [512, 512]
    config.pool_size=50
    config.continue_train=False    

    config.result_dir = "/home/s164435/CycleGAN/result"
    config.ckpt_dir = "/mnt/md0/xiao/ckpt_cyclegan_patchD"

  return config