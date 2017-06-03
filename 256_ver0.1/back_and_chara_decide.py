import numpy as np 
import cv2
import copy
import os.path
import re
import glob
from config import*
from model import*
import model
from mynet_input import*

def InputImageFromFile(image_path,triming=None,resize=None):
    image = cv2.imread(image_path)
    if resize ==True:
        image = cv2.resize(image,(IMAGE_WIDTH,IMAGE_HEIGHT))
    b,g,r = cv2.split(image)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb

    
    return rgb_img

def prediction_test():
    if not os.path.exists('background'):
        os.mkdir('background')
    if not os.path.exists('character'):
        os.mkdir('character')
    with tf.Graph().as_default():
        rgb_image = tf.placeholder(dtype = tf.float32,shape=(IMAGE_HEIGHT,IMAGE_WIDTH,CHANNEL))
        #input_image = tf.constant(value=rgb_image,shape=(1,IMAGE_HEIGHT,IMAGE_WIDTH,CHANNEL),dtype=tf.float32)
        input_image = tf.cast(rgb_image, tf.float32) * (1. / 255) - 0.5
        input_image = tf.reshape(input_image,[1,IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        logits = model.vgg(image=input_image,training_phase=False)

        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess,'log/mytrain/model.ckpt-19500')
        
        inputs = InputImageFromFile(image_path='dataset/000002576.png',resize=True)
        logits_value=sess.run([logits],feed_dict={rgb_image: inputs})
        print(logits_value)  

if __name__ == '__main__':
    prediction_test()
        