# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
from skimage import color,filters
import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument('--save_VIS_dir', dest='save_VIS_dir', default='./results/Fusion/VIS_enhance/', help='directory for testing outputs')
parser.add_argument('--save_Fuse_dir', dest='save_Fuse_dir', default='./results/Fusion/fusion/', help='directory for testing outputs')
parser.add_argument('--test_IR_dir', dest='test_IR_dir', default='./Dataset/Test/Fusion/IR/', help='directory for testing inputs')
parser.add_argument('--test_VIS_dir', dest='test_VIS_dir', default='./Dataset/Test/Fusion/VIS/', help='directory for testing inputs')
parser.add_argument('--ratio', dest='ratio', default=1.5, help='ratio for illumination adjustment')


args = parser.parse_args()
sess = tf.Session()
training = tf.placeholder_with_default(False, shape=(), name='training')
input_IR = tf.placeholder(tf.float32, [None, None, None, 1], name='input_IR')
input_VIS = tf.placeholder(tf.float32, [None, None, None, 3], name='input_VIS')
enhance_ratio = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i_ratio')

VIS_enhance =  Enhance_net(input_VIS,enhance_ratio,training=False)
Fusion_image = Fusion_net(VIS_enhance,input_IR,training=False)


# load pretrained model
var_Enhance = [var for var in tf.trainable_variables() if 'Enhance_net' in var.name]
var_Fuse = [var for var in tf.trainable_variables() if 'Fusion_net' in var.name]
g_list = tf.global_variables()

saver_Enhance = tf.train.Saver(var_list = var_Enhance)
saver_Fuse = tf.train.Saver(var_list = var_Fuse)

enhance_checkpoint_dir ='./checkpoint/Enhance_net/'
enhance_ckpt_pre=tf.train.get_checkpoint_state(enhance_checkpoint_dir)
if enhance_ckpt_pre:
    print('loaded '+enhance_ckpt_pre.model_checkpoint_path)
    saver_Enhance.restore(sess,enhance_ckpt_pre.model_checkpoint_path)
else:
    print('No enhance checkpoint!')


fuse_checkpoint_dir ='./checkpoint/Fusion_net/'
fuse_ckpt_pre=tf.train.get_checkpoint_state(fuse_checkpoint_dir)
if fuse_ckpt_pre:
    print('loaded '+fuse_ckpt_pre.model_checkpoint_path)
    saver_Fuse.restore(sess,fuse_ckpt_pre.model_checkpoint_path)
else:
    print('No fuse checkpoint!')

save_VIS_dir = args.save_VIS_dir
if not os.path.isdir(save_VIS_dir):
    os.makedirs(save_VIS_dir)
    
save_Fuse_dir = args.save_Fuse_dir
if not os.path.isdir(save_Fuse_dir):
    os.makedirs(save_Fuse_dir)
    
 ###load eval data
eval_IR_data = []
eval_IR_img_name =[]
eval_VIS_data = []
eval_VIS_img_name =[]
eval_IR_data_name = glob(args.test_IR_dir+'*')
eval_IR_data_name.sort()
eval_VIS_data_name = glob(args.test_VIS_dir+'*')
eval_VIS_data_name.sort()

for idx in range(len(eval_IR_data_name)):
    [_, name]  = os.path.split(eval_IR_data_name[idx])        
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_IR_img_name.append(name)
    eval_IR_im = load_images(eval_IR_data_name[idx])
    eval_IR_im = np.expand_dims(eval_IR_im,2)
    eval_VIS_im = load_images(eval_VIS_data_name[idx])
    h,w,c = eval_IR_im.shape
    #print(eval_IR_im.shape)
    h_tmp = h%1
    w_tmp = w%1
    eval_IR_im_resize = eval_IR_im[0:h-h_tmp, 0:w-w_tmp, :]
    eval_VIS_im_resize = eval_VIS_im[0:h-h_tmp, 0:w-w_tmp, :]
    eval_IR_data.append(eval_IR_im_resize)
    eval_VIS_data.append(eval_VIS_im_resize)


print("Start evalating!")
start_time = time.time()
for idx in range(len(eval_IR_data)):
    print(idx)
    name = eval_IR_img_name[idx]
    input_IR_im = eval_IR_data[idx]
    input_VIS_im = eval_VIS_data[idx]    
    input_IR_eval = np.expand_dims(input_IR_im, axis=0)
    input_VIS_eval = np.expand_dims(input_VIS_im, axis=0)
    h_ratio,w_ratio,c = eval_IR_data[idx].shape
    ratio = float(args.ratio)
    enhance_set_ratio = np.ones([h_ratio, w_ratio])*(ratio)
    #print(enhance_set_ratio.shape)
    enhance_set_ratio_expand = np.expand_dims(enhance_set_ratio , axis =2)
    enhance_set_ratio_expand2 = np.expand_dims(enhance_set_ratio_expand, axis=0)
    

    enhance_vis_imag,fusion_image = sess.run([VIS_enhance,Fusion_image], feed_dict={input_IR: input_IR_eval,input_VIS: input_VIS_eval,enhance_ratio: enhance_set_ratio_expand2})

    save_images(os.path.join(save_Fuse_dir, '%s.png' % (name)), fusion_image)
    save_images(os.path.join(save_VIS_dir, '%s.png' % (name)), enhance_vis_imag)


    
    
