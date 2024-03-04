# coding: utf-8
from __future__ import print_function
import os, time, random
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import *
from model import *
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--train_data_dir', dest='train_data_dir', default='./Dataset/Training/', help='directory for training inputs')

args = parser.parse_args()

batch_size = args.batch_size
patch_size = args.patch_size

sess = tf.Session()


############### placeholder for input ###################
input_IR = tf.placeholder(tf.float32, [None, None, None, 1], name='input_IR')
input_VIS = tf.placeholder(tf.float32, [None, None, None, 3], name='input_VIS') 
input_Ratio = tf.placeholder(tf.float32, [None, None, None, 1], name='input_Ratio')

input_IR_3 = tf.concat([input_IR, input_IR, input_IR],3)


############### Run EnhanceNet and  FusionNet ###################
VIS_enhance  = Enhance_net(input_VIS,input_Ratio)
Fuse_img  = Fusion_net(VIS_enhance, input_IR)


############### get the single-channel version ###################
input_VIS_yuv = tf.image.rgb_to_yuv(input_VIS)
input_VIS_gray = tf.expand_dims(input_VIS_yuv[:,:,:,0],-1)

VIS_enhance_yuv = tf.image.rgb_to_yuv(VIS_enhance)
VIS_enhance_gray = tf.expand_dims(VIS_enhance_yuv[:,:,:,0],-1)

Fusion_yuv =  tf.image.rgb_to_yuv(Fuse_img)
Fusion_gray = tf.expand_dims(Fusion_yuv[:,:,:,0],-1)


############### get the required low-pass version ###################
input_IR_blur = blur(input_IR)
input_VIS_gray_blur=blur(input_VIS_gray)
VIS_enhance_gray_blur=blur(VIS_enhance_gray)
############### get the rough reflectance maps ###################
input_VIS_scene=tf.div(input_VIS,tf.maximum(input_VIS_gray_blur,0.001))
VIS_enhance_scene=tf.div(VIS_enhance,tf.maximum(VIS_enhance_gray_blur,0.001))


############### get the required high-pass version ###################
input_VIS_R = tf.expand_dims(input_VIS[:,:,:,0],-1)
input_VIS_G = tf.expand_dims(input_VIS[:,:,:,1],-1)
input_VIS_B = tf.expand_dims(input_VIS[:,:,:,2],-1)
input_VIS_R_grad_x = gradient((input_VIS_R), "x")
input_VIS_R_grad_y = gradient((input_VIS_R), "y")
input_VIS_R_grad = tf.abs(input_VIS_R_grad_x)+tf.abs(input_VIS_R_grad_y)
input_VIS_G_grad_x = gradient((input_VIS_G), "x")
input_VIS_G_grad_y = gradient((input_VIS_G), "y")
input_VIS_G_grad = tf.abs(input_VIS_G_grad_x)+tf.abs(input_VIS_G_grad_y)
input_VIS_B_grad_x = gradient((input_VIS_B), "x")
input_VIS_B_grad_y = gradient((input_VIS_B), "y")
input_VIS_B_grad = tf.abs(input_VIS_B_grad_x)+tf.abs(input_VIS_B_grad_y)
input_VIS_grad = tf.concat([input_VIS_R_grad, input_VIS_G_grad,input_VIS_B_grad], 3)


input_IR_grad_x = gradient((input_IR), "x")
input_IR_grad_y = gradient((input_IR), "y")
input_IR_grad = tf.abs(input_IR_grad_x)+tf.abs(input_IR_grad_y)
input_IR_3_grad = tf.concat([input_IR_grad, input_IR_grad,input_IR_grad], 3)

input_en_VIS_R = tf.expand_dims(VIS_enhance[:,:,:,0],-1)
input_en_VIS_G = tf.expand_dims(VIS_enhance[:,:,:,1],-1)
input_en_VIS_B = tf.expand_dims(VIS_enhance[:,:,:,2],-1)
enhnace_VIS_R_grad_x = gradient((input_en_VIS_R), "x")
enhnace_VIS_R_grad_y = gradient((input_en_VIS_R), "y")
enhnace_VIS_R_grad = tf.abs(enhnace_VIS_R_grad_x)+tf.abs(enhnace_VIS_R_grad_y)
enhnace_VIS_G_grad_x = gradient((input_en_VIS_G), "x")
enhnace_VIS_G_grad_y = gradient((input_en_VIS_G), "y")
enhnace_VIS_G_grad = tf.abs(enhnace_VIS_G_grad_x)+tf.abs(enhnace_VIS_G_grad_y)
enhnace_VIS_B_grad_x = gradient((input_en_VIS_B), "x")
enhnace_VIS_B_grad_y = gradient((input_en_VIS_B), "y")
enhnace_VIS_B_grad = tf.abs(enhnace_VIS_B_grad_x)+tf.abs(enhnace_VIS_B_grad_y)
enhnace_VIS_grad = tf.concat([enhnace_VIS_R_grad, enhnace_VIS_G_grad,enhnace_VIS_B_grad], 3)


Fused_image_R = tf.expand_dims(Fuse_img[:,:,:,0],-1)
Fused_image_G = tf.expand_dims(Fuse_img[:,:,:,1],-1)
Fused_image_B = tf.expand_dims(Fuse_img[:,:,:,2],-1)
Fused_image_R_grad_x = gradient((Fused_image_R), "x") 
Fused_image_R_grad_y = gradient((Fused_image_R), "y") 
Fused_image_R_grad = tf.abs(Fused_image_R_grad_x)+tf.abs(Fused_image_R_grad_y)
Fused_image_G_grad_x = gradient((Fused_image_G), "x") 
Fused_image_G_grad_y = gradient((Fused_image_G), "y") 
Fused_image_G_grad = tf.abs(Fused_image_G_grad_x)+tf.abs(Fused_image_G_grad_y)
Fused_image_B_grad_x = gradient((Fused_image_B), "x") 
Fused_image_B_grad_y = gradient((Fused_image_B), "y") 
Fused_image_B_grad = tf.abs(Fused_image_B_grad_x)+tf.abs(Fused_image_B_grad_y)
Fused_image_grad = tf.concat([Fused_image_R_grad, Fused_image_G_grad,Fused_image_B_grad], 3)


############### get the fusion target ###################
## generate the targeted gradient 
joint_grad = tf.maximum(enhnace_VIS_grad,input_IR_3_grad)
## generate the targeted intensity 
joint_int = tf.maximum(VIS_enhance,input_IR_3)

############### get the color vector ###################
VIS_enhance_color_V = tf.nn.l2_normalize(VIS_enhance,axis=3) 
Fuse_color_V = tf.nn.l2_normalize(Fuse_img,axis=3)
IR_color_V = tf.nn.l2_normalize(input_IR_3,axis=3)


#############  LOSS FUNCTION ######################

##### EnhanceNet Loss #####
#Discriminator
True_prob=E_discriminator(input_IR_blur,input_Ratio)
Fake_prob=E_discriminator(VIS_enhance_gray_blur,input_Ratio)
#discriminator loss
Dis_loss_true=tf.reduce_mean(tf.abs(True_prob-tf.random_uniform(shape=[batch_size,1],minval=0.8,maxval=1.0,dtype=tf.float32)))
Dis_loss_fake=tf.reduce_mean(tf.abs(Fake_prob-tf.random_uniform(shape=[batch_size,1],minval=0.0,maxval=0.2,dtype=tf.float32)))
Dis_loss_total=Dis_loss_true+Dis_loss_fake
tf.summary.scalar('Dis_loss_total',Dis_loss_total)

##generator loss
#FIDELITY LOSS
VIS_fidelity_loss=tf.reduce_mean(tf.abs(input_VIS_scene-VIS_enhance_scene)) 
fidelity_loss =  1*VIS_fidelity_loss
#Enhance_dis LOSS 
G_loss_dis=tf.reduce_mean(tf.abs(Fake_prob-tf.random_uniform(shape=[batch_size,1],minval=0.8,maxval=1.0,dtype=tf.float32)))
#Generator total loss
G_loss_total =  10*fidelity_loss + 1*G_loss_dis 
tf.summary.scalar('fidelity_loss',fidelity_loss)
tf.summary.scalar('G_loss_dis',G_loss_dis)


###### FusionNet Loss #####
# Gradient loss for texture preservation
Fusion_grad_loss = tf.reduce_mean(tf.abs(Fused_image_grad-joint_grad))

## Intensity loss for salient information integration
Fusion_int_loss = tf.reduce_mean(tf.abs(Fuse_img-joint_int))

Fusion_color_loss = tf.reduce_mean((1 - tf.reduce_sum(Fuse_color_V * VIS_enhance_color_V, axis=3, keepdims=True)))


Fusion_loss_total = 1*Fusion_int_loss+ 1*Fusion_color_loss + 10*Fusion_grad_loss 
tf.summary.scalar('Fusion_grad_loss',Fusion_grad_loss)
tf.summary.scalar('Fusion_int_loss',Fusion_int_loss)
tf.summary.scalar('Fusion_color_loss',Fusion_color_loss)


lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
var_Fuse = [var for var in tf.trainable_variables() if 'Fusion_net' in var.name]
var_Enhance = [var for var in tf.trainable_variables() if 'Enhance_net' in var.name]
var_E_Dis = [var for var in tf.trainable_variables() if 'E_discriminator' in var.name]
g_list = tf.global_variables()

with tf.name_scope('train_step'):
  train_op_Fuse = optimizer.minimize(Fusion_loss_total, var_list = var_Fuse)
  train_op_Enhance = optimizer.minimize(G_loss_total, var_list = var_Enhance)
  train_op_E_Dis = optimizer.minimize(Dis_loss_total, var_list = var_E_Dis)

saver_fuse = tf.train.Saver(var_list=var_Fuse,max_to_keep=2000)
saver_enhance = tf.train.Saver(var_list=var_Enhance,max_to_keep=2000)
saver_E_Dis = tf.train.Saver(var_list=var_E_Dis,max_to_keep=2000)
sess.run(tf.global_variables_initializer())

print("[*] Initialize model successfully...")

with tf.name_scope('image'):
  tf.summary.image('input_IR',tf.expand_dims(input_IR[1,:,:,:],0))
  tf.summary.image('input_VIS',tf.expand_dims(input_VIS[1,:,:,:],0)) 
  tf.summary.image('VIS_enhance',tf.expand_dims(VIS_enhance[1,:,:,:],0))
  tf.summary.image('input_VIS_scene',tf.expand_dims(input_VIS_scene[1,:,:,:],0))
  tf.summary.image('VIS_enhance_scene',tf.expand_dims(VIS_enhance_scene[1,:,:,:],0))
  tf.summary.image('Fuse_img',tf.expand_dims(Fuse_img[1,:,:,:],0))
  
summary_op = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./log' + '/DDBF_train',sess.graph,flush_secs=60)


#load data
###train_data
train_IR_1_data = []
train_IR_2_data = []
train_IR_3_data = []
train_VIS_data  = []

train_IR_1_data_names = glob(args.train_data_dir +'/night/IR_1/*.png') 
train_IR_1_data_names.sort()
train_IR_2_data_names = glob(args.train_data_dir +'/night/IR_2/*.png') 
train_IR_2_data_names.sort()
train_IR_3_data_names = glob(args.train_data_dir +'/night/IR_3/*.png') 
train_IR_3_data_names.sort()
train_VIS_data_names = glob(args.train_data_dir +'/night/VIS/*.png') 
train_VIS_data_names.sort()

assert len(train_IR_1_data_names) == len(train_VIS_data_names)
print('[*] Number of training data: %d' % len(train_IR_1_data_names))
for idx in range(len(train_IR_1_data_names)):
    IR_1_im = load_images(train_IR_1_data_names[idx])
    IR_1_im = np.expand_dims(IR_1_im,2)
    train_IR_1_data.append(IR_1_im)
    IR_2_im = load_images(train_IR_2_data_names[idx])
    IR_2_im = np.expand_dims(IR_2_im,2) 
    train_IR_2_data.append(IR_2_im)   
    IR_3_im = load_images(train_IR_3_data_names[idx])
    IR_3_im = np.expand_dims(IR_3_im,2) 
    train_IR_3_data.append(IR_3_im)              
    VIS_im = load_images(train_VIS_data_names[idx])
    train_VIS_data.append(VIS_im)



epoch = 1500
learning_rate = 0.0001
train_phase = 'DDBF'
numBatch = len(train_IR_1_data) // int(batch_size)


enhance_checkpoint_dir = './checkpoint/Enhance_net/'
fuse_checkpoint_dir = './checkpoint/Fusion_net/'

if not os.path.isdir(enhance_checkpoint_dir):
    os.makedirs(enhance_checkpoint_dir)
ckpt_enhance=tf.train.get_checkpoint_state(enhance_checkpoint_dir)
if ckpt_enhance:
    print('loaded '+ckpt_enhance.model_checkpoint_path)
    saver_enhance.restore(sess,ckpt_enhance.model_checkpoint_path)
else:
    print('No Enhance pretrained model!')


if not os.path.isdir(fuse_checkpoint_dir):
    os.makedirs(fuse_checkpoint_dir)
ckpt_fusion=tf.train.get_checkpoint_state(fuse_checkpoint_dir)
if ckpt_fusion:
    print('loaded '+ckpt_fusion.model_checkpoint_path)
    saver_fuse.restore(sess,ckpt_fusion.model_checkpoint_path)
else:
    print('No Fusion pretrained model!')

start_step = 0
start_epoch = 0
iter_num = 0

print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
start_time = time.time()
image_id = 0
counter = 0
for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):
        batch_input_IR_1 = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_IR_2 = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_IR_3 = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")      
        batch_input_VIS = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        
        batch_IR_1_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_IR_2_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_IR_3_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")


        Rand_input_IR = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")        
        Rand_input_VIS = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        Rand_input_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        
        
        for patch_id in range(batch_size):
                        
            h, w,_ = train_IR_1_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            
            rand_mode = random.randint(0, 7)            
            train_IR_1_data_crop = train_IR_1_data[image_id][x : x+patch_size, y : y+patch_size,:]
            train_IR_2_data_crop = train_IR_2_data[image_id][x : x+patch_size, y : y+patch_size,:]
            train_IR_3_data_crop = train_IR_3_data[image_id][x : x+patch_size, y : y+patch_size,:]
                                              
            train_VIS_data_crop = train_VIS_data[image_id][x : x+patch_size, y : y+patch_size, :]
                        
            batch_input_IR_1[patch_id, :, :,:] = data_augmentation(train_IR_1_data_crop, rand_mode)
            batch_input_IR_2[patch_id, :, :,:] = data_augmentation(train_IR_2_data_crop, rand_mode)
            batch_input_IR_3[patch_id, :, :,:] = data_augmentation(train_IR_3_data_crop, rand_mode)
            
            batch_input_VIS[patch_id, :, :, :] = data_augmentation(train_VIS_data_crop, rand_mode)
            
            ratio_1 = 0.2
            IR_1_ratio = np.ones([patch_size,patch_size])*(ratio_1)
            IR_1_ratio_expand = np.expand_dims(IR_1_ratio , axis =2)
            batch_IR_1_ratio[patch_id, :, :, :] = IR_1_ratio_expand
            
            ratio_2 = 1
            IR_2_ratio = np.ones([patch_size,patch_size])*(ratio_2)
            IR_2_ratio_expand = np.expand_dims(IR_2_ratio , axis =2)
            batch_IR_2_ratio[patch_id, :, :, :] = IR_2_ratio_expand            
                       
            ratio_3 = 5.0
            IR_3_ratio = np.ones([patch_size,patch_size])*(ratio_3)
            IR_3_ratio_expand = np.expand_dims(IR_3_ratio , axis =2)
            batch_IR_3_ratio[patch_id, :, :, :] = IR_3_ratio_expand             
                        
                    
            rand_ratio_mode = np.random.randint(0, 3)
            if rand_ratio_mode == 1:            
              Rand_input_IR[patch_id, :, :, :] = batch_input_IR_1[patch_id, :, :, :]
              Rand_input_VIS[patch_id, :, :, :] = batch_input_VIS[patch_id, :, :, :]
              Rand_input_ratio[patch_id, :, :, :] = batch_IR_1_ratio[patch_id, :, :, :]
            elif rand_ratio_mode == 2:
              Rand_input_IR[patch_id, :, :, :] = batch_input_IR_2[patch_id, :, :, :]
              Rand_input_VIS[patch_id, :, :, :] = batch_input_VIS[patch_id, :, :, :]
              Rand_input_ratio[patch_id, :, :, :] = batch_IR_2_ratio[patch_id, :, :, :]
            elif rand_ratio_mode == 3:
              Rand_input_IR[patch_id, :, :, :] = batch_input_IR_3[patch_id, :, :, :]
              Rand_input_VIS[patch_id, :, :, :] = batch_input_VIS[patch_id, :, :, :]
              Rand_input_ratio[patch_id, :, :, :] = batch_IR_3_ratio[patch_id, :, :, :]
            
            image_id = (image_id + 1) % len(train_IR_1_data)

                                
        for i in range(2):
            _, loss_dis = sess.run([train_op_E_Dis, Dis_loss_total], feed_dict={input_IR: Rand_input_IR,input_VIS: Rand_input_VIS,input_Ratio: Rand_input_ratio,lr: learning_rate})
            _, loss_enhance = sess.run([train_op_Enhance, G_loss_total], feed_dict={input_IR: Rand_input_IR,input_VIS: Rand_input_VIS,input_Ratio: Rand_input_ratio,lr: learning_rate})
        _, loss_fusion, summary_str = sess.run([train_op_Fuse, Fusion_loss_total,summary_op], feed_dict={input_IR: Rand_input_IR,input_VIS: Rand_input_VIS,input_Ratio: Rand_input_ratio,lr: learning_rate})
            
        if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d],  loss_d: [%.8f],loss_g:[%.8f],loss_f:[%.8f]" \
              % ((epoch+1), iter_num, loss_dis,loss_enhance,loss_fusion))
        train_writer.add_summary(summary_str,iter_num)
        iter_num += 1

        
    global_step = epoch+1
    if (epoch+1)%10==0:
      saver_enhance.save(sess, enhance_checkpoint_dir + 'model.ckpt', global_step=global_step)
      saver_fuse.save(sess, fuse_checkpoint_dir + 'model.ckpt', global_step=global_step)

print("[*] Finish training for phase %s." % train_phase)
