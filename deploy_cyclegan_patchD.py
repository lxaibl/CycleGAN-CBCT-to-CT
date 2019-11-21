#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:57:37 2018

@author: s164435
"""

import tensorflow as tf
from models_cyclegan_patchD import CycleGAN
from config_cyclegan_patchD import get_config
from ops_cyclegan_patchD import mkdir
import os
import numpy as np
import matplotlib.pyplot as plt
import glob


def fetch_training_data_dirs(data_dir):
    training_data_dirs = list()
    for subject in glob.glob(os.path.join(data_dir, "*")):
        training_data_dirs.append(subject)
    return training_data_dirs

def main():
  sess = tf.Session()
  config = get_config(is_train=False)
  mkdir(config.result_dir)

  reg = CycleGAN(sess, config, "CycleGAN", is_train=False)
  reg.restore(config.ckpt_dir)

  cbct_dirs=fetch_training_data_dirs('U:/s164435/My Documents/data/CBCT to CT/HN/Testing/CBCT_raw/')
  dpct_dirs=fetch_training_data_dirs('U:/s164435/My Documents/data/CBCT to CT/HN/Testing/dpCT_raw/')

  for i in range(len(cbct_dirs)):
      sct=[]
      #read cbct raw data
      fname = cbct_dirs[i]
      with open(fname, 'r') as infile:
          infile.seek(0,0)
          data = np.fromfile(infile, dtype='f') 
      if data.shape[0]%(512*512)==0:
          Nx=512
      else: 
          Nx=410
      cbct=data.reshape(int(data.shape[0]/Nx/Nx),Nx,Nx) #cbct volume
      cbct=np.transpose(cbct,(2,1,0))
      cbct=cbct[:,:,7:87] # select slice 7 to slice 87
      if Nx==410:
          cbct=np.pad(cbct,((51,51),(51,51),(0,0)),'constant', constant_values=-1000.0)
      
      #read dpct raw data
      #fname = dpct_dirs[i]  #for HN
      fname = dpct_dirs[i*2+1] #prostate i*2+1
      with open(fname, 'r') as infile:
          infile.seek(0,0)
          data = np.fromfile(infile, dtype='f') 
      if data.shape[0]%(512*512)==0:
          Nx=512
      else: 
          Nx=410
      dpct=data.reshape(int(data.shape[0]/Nx/Nx),Nx,Nx) #ct volume
      dpct=np.transpose(dpct,(2,1,0))
      dpct=dpct[:,:,7:87] # select slice 7 to slice 87
      if Nx==410:
          dpct=np.pad(dpct,((51,51),(51,51),(0,0)),'constant', constant_values=-1000.0)

      #create a folder to save images
      patientID=fname.split('\\')[1].split('_')[0]
      image_result_dir = "T:/Physics Research/Users/Xiao Liang/CycleGAN Prostate Test/"+patientID
      mkdir(image_result_dir)
      
      #test model
      #Remeber to normalize data from HU unit before testing.
      for ii in range(cbct.shape[2]):
          batch_dpct=dpct[:,:,ii]
          batch_dpct=np.tanh(batch_dpct/1000)
          batch_dpct=batch_dpct[np.newaxis,:,:,np.newaxis]
          
          batch_cbct =cbct[:,:,ii]
          batch_cbct=np.where(batch_cbct>=1500,(batch_cbct-1500)*(3071-1500)/5500+1500,batch_cbct)
          batch_cbct=np.tanh(batch_cbct/1000)
          batch_cbct=batch_cbct[np.newaxis,:,:,np.newaxis]
          
          image_result_dir1=image_result_dir+'/{:0>3d}'.format(ii)
          mkdir(image_result_dir1)
          synct=reg.deploy(image_result_dir1,batch_cbct,batch_dpct)
          synct=synct[0,:,:,0]
          #normalize synct back to HU unit.
          synct=np.arctanh(synct)*1000
          synct=np.clip(synct,-1000,3071)
          sct.append(synct[:,:])
         
      #save sct volume to raw data file
      sct=np.asarray(sct)
      sct=np.transpose(sct,(0,2,1))
      sct1D=sct[:,:,:].reshape(-1)
      sct_dir="U:/s164435/My Documents/data/CBCT to CT/Prostate/Testing/sCT_raw/"
      fd = open(sct_dir+patientID+'_sct.raw', 'wb')
      sct1D.tofile(fd)
      fd.close()

if __name__ == "__main__":
  main()

