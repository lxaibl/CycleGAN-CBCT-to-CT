#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:42:52 2018

@author: s164435
"""

import tensorflow as tf
from tensorflow.contrib.keras.api.keras.layers import UpSampling2D
from ops_cyclegan_patchD import *

class Generator(object):
  def __init__(self, name,batch_size, is_train,reuse):
    self.name = name
    self.is_train = is_train
    self.reuse = reuse
  
  def __call__(self, x):
    with tf.variable_scope(self.name, reuse=self.reuse):
      conv1 = conv2d(x, "conv1", 32, 4, 1, "SAME", True, True, self.is_train)# 512*512*32
      conv1 = conv2d(conv1, "conv2", 32, 4, 2, "SAME", True, True, self.is_train)#256*256*32
      
      conv2 = conv2d(conv1, "conv3", 64, 4, 1, "SAME", True, True, self.is_train)#256*256*64
      conv2 = conv2d(conv2, "conv4", 64, 4, 2, "SAME", True, True, self.is_train)#128*128*64

      conv3 = conv2d(conv2, "conv5", 128, 4, 1, "SAME", True, True, self.is_train)#128*128*128
      conv3 = conv2d(conv3, "conv6", 128, 4, 2, "SAME", True, True, self.is_train)#64*64*128

      conv4 = conv2d(conv3, "conv7", 256, 4, 1, "SAME", True, True, self.is_train)#64*64*256
      conv4 = conv2d(conv4, "conv8", 256, 4, 2, "SAME", True, True, self.is_train)#32*32*256
      
      
      conv5 = conv2d(conv4, "conv9", 512, 4, 1, "SAME", True, True, self.is_train)#32*32*512
      conv5 = conv2d(conv5, "conv10", 512, 4, 2, "SAME", True, True, self.is_train)#16*16*512
      
      up5 = UpSampling2D((2,2))(conv5)#32*32*512
      up5 = conv2d(up5, "conv15", 256, 4, 1, "SAME", True, True, self.is_train)#32*32*256
      up5 = tf.concat([up5,conv4],axis=3)
      up5 = conv2d(up5, "conv16", 256, 4, 1, "SAME", True, True, self.is_train)#32*32*256
      
      
      up4 = UpSampling2D((2,2))(up5)#64*64*256
      up4 = conv2d(up4, "conv17", 128, 4, 1, "SAME", True, True, self.is_train)#64*64*128
      up4 = tf.concat([up4,conv3],axis=3)
      up4 = conv2d(up4, "conv18", 128, 4, 1, "SAME", True, True, self.is_train)#64*64*128
      
      up3 = UpSampling2D((2,2))(up4)#128*128*128
      up3 = conv2d(up3, "conv19", 64, 4, 1, "SAME", True, True, self.is_train)#128*128*64
      up3 = tf.concat([up3,conv2],axis=3)
      up3 = conv2d(up3, "conv20", 64, 4, 1, "SAME", True, True, self.is_train)#129*128*64
      
      up2 = UpSampling2D((2,2))(up3)#256*256*64
      up2 = conv2d(up2, "conv21", 32, 4, 1, "SAME", True, True, self.is_train)#256*256*32
      up2 = tf.concat([up2,conv1],axis=3)
      up2 = conv2d(up2, "conv22", 32, 4, 1, "SAME", True, True, self.is_train)#256*256*32
      
      up1 = UpSampling2D((2,2))(up2)#512*512*32
      up1 = conv2d(up1, "conv23", 32, 4, 1, "SAME", True, True, self.is_train)#512*512*32
      up1 = conv2d(up1, "conv24", 1, 1, 1, "SAME", False, False, self.is_train)#512*512*1

      output=tf.nn.tanh(up1)

    if self.reuse is None:
      self.var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      self.saver = tf.train.Saver(self.var_list)
      self.reuse = True

    return output

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)



class Discriminator(object):
  
  def __init__(self, name, batch_size,is_train,reuse):
    self.name = name
    self.is_train = is_train
    self.reuse = reuse
    self.batch_size=batch_size

  def __call__(self, x):
    with tf.variable_scope(self.name,reuse=self.reuse):
      # image is 512 x 512 x 1  
      x=Dc_conv2d(x,"conv1",64,4,2,"SAME",False,True,self.is_train)
      #  (256 x 256 x 64)
      
      x=Dc_conv2d(x,"conv2",128,4,2,"SAME",True,True,self.is_train)
      # (128 x 128 x 128)
      
      x=Dc_conv2d(x,"conv3",256,4,2,"SAME",True,True,self.is_train)
      # (64x 64x 256)
      
      x=Dc_conv2d(x,"conv4",512,4,2,"SAME",True,True,self.is_train)
      # (32 32 512)
      
      x=Dc_conv2d(x,"conv5",512,4,1,"SAME",True,True,self.is_train)
      #32 32 512
          
      x=Dc_conv2d(x,"conv6",1,4,1,"SAME",False,False,self.is_train)
      #32 32 1
      
    if self.reuse is None:
      self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      self.saver = tf.train.Saver(self.var_list)
      self.reuse = True
    
    return x
    
  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)



class CycleGAN(object):
  def __init__(self, sess, config, name, is_train):
    self.sess = sess
    self.name = name
    self.is_train = is_train
    self.pool_ct = ImagePool(config.pool_size,config.im_size[0],config.im_size[1],1)
    self.pool_cbct = ImagePool(config.pool_size,config.im_size[0],config.im_size[1],1)
    
    # moving / fixed images
    im_shape = [config.batch_size] + config.im_size + [1]    
    self.cbct = tf.placeholder(tf.float32, im_shape, name='real_A') #A
    self.ct=tf.placeholder(tf.float32, im_shape, name='real_B')     #B
    self.fake_A_sample = tf.placeholder(tf.float32,im_shape, name='fake_A_sample')
    self.fake_B_sample = tf.placeholder(tf.float32,im_shape, name='fake_B_sample')
    self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
    
    self.generator_A2B=Generator("generator_A2B",config.batch_size, is_train=self.is_train,reuse=None) #cbct to ct
    self.generator_B2A=Generator("generator_B2A",config.batch_size, is_train=self.is_train,reuse=None) #ct to cbct
    self.discriminator_B=Discriminator("discriminator_B",config.batch_size,is_train=self.is_train,reuse=None) # discriminate CT
    self.discriminator_A=Discriminator("discriminator_A",config.batch_size,is_train=self.is_train,reuse=None) # discriminate CBCT

    
    # A to B train
    self.ct_fake = self.generator_A2B(self.cbct)
    self.cbct_fake_ = self.generator_B2A(self.ct_fake)
    self.cbct_fake = self.generator_B2A(self.ct)
    self.ct_fake_ = self.generator_A2B(self.cbct_fake)
    
    self.DB_fake=self.discriminator_B(self.ct_fake)
    self.DA_fake = self.discriminator_A(self.cbct_fake)
    
    self.DB_real=self.discriminator_B(self.ct)
    self.DA_real=self.discriminator_A(self.cbct)
    self.DB_fake_sample=self.discriminator_B(self.fake_B_sample)
    self.DA_fake_sample=self.discriminator_A(self.fake_A_sample)
           
    self.ct_identity=self.generator_A2B(self.ct)
    self.cbct_identity=self.generator_B2A(self.cbct)
    
    if self.is_train :
      #training loss
      self.g_loss_mse_DB=mse(self.DB_fake, tf.ones_like(self.DB_fake))
      self.g_loss_mse_DA=mse(self.DA_fake,tf.ones_like(self.DA_fake))
      self.g_loss_mae_cbct=config.weight*mae(self.cbct,self.cbct_fake_)
      self.g_loss_mae_ct=config.weight*mae(self.ct,self.ct_fake_)
      self.g_loss_mae_cbct_identity=config.weight_identity*mae(self.cbct,self.cbct_identity)
      self.g_loss_mae_ct_identity=config.weight_identity*mae(self.ct,self.ct_identity)
      self.g_loss=self.g_loss_mse_DB+self.g_loss_mse_DA+self.g_loss_mae_cbct+self.g_loss_mae_ct+self.g_loss_mae_cbct_identity+self.g_loss_mae_ct_identity
      
      self.db_loss_real = mse(self.DB_real, tf.ones_like(self.DB_real))
      self.db_loss_fake_sample = mse(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
      self.db_loss = (self.db_loss_real + self.db_loss_fake_sample) / 2
      self.da_loss_real = mse(self.DA_real, tf.ones_like(self.DA_real))
      self.da_loss_fake_sample = mse(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
      self.da_loss = (self.da_loss_real + self.da_loss_fake_sample) / 2
      self.d_loss = self.da_loss + self.db_loss
      
      
      #validation loss
      self.db_loss_fake=mse(self.DB_fake, tf.zeros_like(self.DB_fake))
      self.validation_db_loss = (self.db_loss_real + self.db_loss_fake) / 2
      self.da_loss_fake = mse(self.DA_fake, tf.zeros_like(self.DA_fake))
      self.validation_da_loss = (self.da_loss_real + self.da_loss_fake) / 2
      self.validation_d_loss=self.validation_db_loss+self.validation_da_loss
      
      
      #training summary
      self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
      self.d_loss_sum=tf.summary.scalar("d_loss", self.d_loss)

      #validating summary
      self.validation_g_loss_sum=tf.summary.scalar("validation_g_loss",self.g_loss)
      self.validation_d_loss_sum=tf.summary.scalar("validation_d_loss",self.validation_d_loss)
      
      #minimize loss for generator and discriminator
      self.optim = tf.train.AdamOptimizer(self.lr,beta1=0.5)
      self.g_train = self.optim.minimize(self.g_loss, var_list=[self.generator_A2B.var_list,self.generator_B2A.var_list])
      self.d_train = self.optim.minimize(self.d_loss, var_list=[self.discriminator_B.var_list,self.discriminator_A.var_list])


    self.sess.run(tf.global_variables_initializer())
    self.writer = tf.summary.FileWriter(config.logs, self.sess.graph)
    
    if config.continue_train:
        self.restore(config.ckpt_dir)
        print("Load SUCCESS")


  def fit(self, batch_cbct2, batch_ct2,lr,counter, cbct2_validation=None, ct2_validation=None):
    _, cbct_fake,ct_fake,g_loss_sum=self.sess.run([self.g_train, self.cbct_fake, self.ct_fake,self.g_loss_sum],{self.cbct:batch_cbct2, self.ct:batch_ct2,self.lr:lr})
    self.writer.add_summary(g_loss_sum, counter)
    
    ct_fake = self.pool_ct(ct_fake,ct_fake.shape[0])
    cbct_fake = self.pool_cbct(cbct_fake,cbct_fake.shape[0])
    
    #train and write training loss summary
    _,d_loss_sum=self.sess.run([self.d_train, self.d_loss_sum],{self.cbct:batch_cbct2, self.ct:batch_ct2,self.fake_A_sample:cbct_fake,self.fake_B_sample:ct_fake,self.lr:lr})
    self.writer.add_summary(d_loss_sum, counter)
        
    #write validate summary
    validation_g_loss_sum,validation_d_loss_sum= \
    self.sess.run([self.validation_g_loss_sum,self.validation_d_loss_sum], {self.cbct:cbct2_validation, self.ct:ct2_validation})
    self.writer.add_summary(validation_g_loss_sum, counter)
    self.writer.add_summary(validation_d_loss_sum, counter)

    
  def test(self, dir_path, batch_cbct2, batch_ct2):
    cbct_fake,ct_fake = self.sess.run([self.cbct_fake,self.ct_fake], {self.cbct:batch_cbct2, self.ct:batch_ct2})
    for i in range(batch_cbct2.shape[0]):
      save_image_with_scale(dir_path+"/{:02d}_cbct2.tif".format(i+1), batch_cbct2[i,:,:,0])
      save_image_with_scale(dir_path+"/{:02d}_ct2.tif".format(i+1), batch_ct2[i,:,:,0])
      save_image_with_scale(dir_path+"/{:02d}_ct2_fake.tif".format(i+1), ct_fake[i,:,:,0])
      save_image_with_scale(dir_path+"/{:02d}_cbct2_fake.tif".format(i+1), cbct_fake[i,:,:,0])

  def deploy(self, dir_path, batch_cbct2,batch_ct2):
    cbct_fake,ct_fake = self.sess.run([self.cbct_fake,self.ct_fake], {self.cbct:batch_cbct2, self.ct:batch_ct2})
    for i in range(batch_cbct2.shape[0]):
      save_image_with_scale(dir_path+"/{:02d}_cbct2.tif".format(i+1), batch_cbct2[i,:,:,0])
      save_image_with_scale(dir_path+"/{:02d}_ct2.tif".format(i+1), batch_ct2[i,:,:,0])
      save_image_with_scale(dir_path+"/{:02d}_ct2_fake.tif".format(i+1), ct_fake[i,:,:,0])
      save_image_with_scale(dir_path+"/{:02d}_cbct2_fake.tif".format(i+1), cbct_fake[i,:,:,0])
    return ct_fake

  def save(self, dir_path):
    self.generator_A2B.save(self.sess, dir_path+"/GeneratorA2Bmodel.ckpt")
    self.generator_B2A.save(self.sess, dir_path+"/GeneratorB2Amodel.ckpt")
    self.discriminator_B.save(self.sess, dir_path+"/discriminator_B.ckpt")
    self.discriminator_A.save(self.sess, dir_path+"/discriminator_A.ckpt")

  def restore(self, dir_path):
    self.generator_A2B.restore(self.sess, dir_path+"/GeneratorA2Bmodel.ckpt")
    self.generator_B2A.restore(self.sess, dir_path+"/GeneratorB2Amodel.ckpt")
    self.discriminator_B.restore(self.sess, dir_path+"/discriminator_B.ckpt")
    self.discriminator_A.restore(self.sess, dir_path+"/discriminator_A.ckpt")