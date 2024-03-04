import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from utils import *

def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable= True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)

        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])

        return deconv_output





def Enhance_net(input_vis, input_ratio, training = True):
    with tf.variable_scope('Enhance_net', reuse=tf.AUTO_REUSE):
        input_cat = tf.concat([input_vis, input_ratio], 3)
         
        conv1_vis_1=slim.conv2d(input_cat, 16,[3,3], rate=1, activation_fn=lrelu,scope='En_conv1_vis_1')
        conv1_vis_1_out = tf.concat([conv1_vis_1, input_ratio], 3)
        
        conv1_vis_2=slim.conv2d(conv1_vis_1_out,16,[3,3], rate=1, activation_fn=lrelu,scope='En_conv1_vis_2')
        conv1_vis_2_out = tf.concat([conv1_vis_2, input_ratio], 3)

        conv2_vis_1=slim.conv2d(conv1_vis_2_out,16,[3,3], rate=1, activation_fn=lrelu,scope='En_conv2_vis_1')
        conv2_vis_1_out = tf.concat([conv2_vis_1, input_ratio], 3)
        
        conv2_vis_2=slim.conv2d(conv2_vis_1_out,16,[3,3], rate=1, activation_fn=lrelu,scope='En_conv2_vis_2')
        conv2_vis_2_out = tf.concat([conv2_vis_2, input_ratio], 3)
                
        conv2_vis_2_add = conv2_vis_2_out+conv1_vis_2_out

                        
        conv3_vis=slim.conv2d(conv2_vis_2_add,16,[3,3], rate=1, activation_fn=lrelu,scope='En_conv3_vis')
        conv3_vis_out = tf.concat([conv3_vis, input_ratio], 3)
        
        conv3_vis_add =conv3_vis_out+conv1_vis_1_out


        ##Enhance_head        
        conv4_vis_1=slim.conv2d(conv3_vis_add,8,[3,3], rate=1, activation_fn=lrelu,scope='En_conv4_vis_1')
        conv4_vis_2=slim.conv2d(conv4_vis_1,3,[3,3], rate=1, activation_fn=None, scope='En_conv4_vis_2')
        conv4_vis_out =tf.sigmoid(conv4_vis_2)                                
                                     
        enhance_image = conv4_vis_out
                                                   
        return enhance_image

def Fusion_net(input_vis, input_ir, training = True):
    with tf.variable_scope('Fusion_net', reuse=tf.AUTO_REUSE):
        Fusion_cat = tf.concat([input_vis,input_ir],3)
        Fusion_conv1 = slim.conv2d(Fusion_cat,16,[3,3], rate=1, activation_fn=lrelu, scope='Fusion_conv1')
        Fusion_conv2 = slim.conv2d(Fusion_conv1,16,[3,3], rate=1, activation_fn=lrelu, scope='Fusion_conv2')
        Fusion_cat_12 = tf.concat([Fusion_conv1,Fusion_conv2],3)
        Fusion_conv3 = slim.conv2d(Fusion_cat_12,16,[3,3], rate=1, activation_fn=lrelu, scope='Fusion_conv3')
        Fusion_cat_123 = tf.concat([Fusion_conv1,Fusion_conv2,Fusion_conv3],3)
        Fusion_conv4 =  slim.conv2d(Fusion_cat_123,16,[3,3], rate=1, activation_fn=lrelu, scope='Fusion_conv4')
        Fusion_conv5 = slim.conv2d(Fusion_conv4,8,[3,3], rate=1, activation_fn=lrelu, scope='Fusion_conv5')
        Fusion_conv6 = slim.conv2d(Fusion_conv5,3,[3,3], rate=1, activation_fn=None, scope='Fusion_conv6')
        Fusion_out =tf.sigmoid(Fusion_conv6)
                                     
        return Fusion_out
        
        
    
def E_discriminator(input_img, input_ratio, training = True):
    with tf.variable_scope('E_discriminator', reuse=tf.AUTO_REUSE):
        input_cat= tf.concat([input_img, input_ratio], 3)
        conv1=slim.conv2d(input_cat, 16,[3,3], rate=1, activation_fn=lrelu,scope='Dis_conv1')
        conv1_out = tf.concat([conv1, input_ratio], 3)
        
        conv2=slim.conv2d(conv1_out,64,[3,3], rate=1, activation_fn=lrelu,scope='Dis_conv2')
        conv2_out = tf.concat([conv2, input_ratio], 3)
        
        conv3=slim.conv2d(conv2_out,256,[3,3], rate=1, activation_fn=lrelu,scope='Dis_conv3')
        conv3_out = tf.concat([conv3, input_ratio], 3)
        
        conv4=slim.conv2d(conv3_out,64,[3,3], rate=1, activation_fn=lrelu,scope='Dis_conv4')
        conv4_out = tf.concat([conv4, input_ratio], 3)
                
        conv5 = slim.conv2d(conv4_out,16,[3,3], rate=1, activation_fn=lrelu,scope='Dis_conv5')
        conv5_out = tf.concat([conv5, input_ratio], 3)
        
        conv6 = slim.conv2d(conv5_out,4,[3,3], rate=1, activation_fn=lrelu,scope='Dis_conv6')
        conv7 = slim.conv2d(conv6,1,[3,3], rate=1, scope='Dis_conv7')
        P = tf.expand_dims(tf.reduce_mean(conv7,axis=[1,2,3]),-1)
        return P

