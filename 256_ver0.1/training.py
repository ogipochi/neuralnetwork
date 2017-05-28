from config import*
from mynet_inputs import*
import model

import copy
from datetime import datetime
import os.path
import re
import time

#時間の計測
start = time.time()
#配列表示の省略なしの設定
np.set_printoptions(threshold=np.inf)

#画像から入力生成
#image_path:画像への相対パス
def InputImageFromFile(image_path):
    #画像の読み込み
    image = cv2.imread(image_path)
    #BGRからRGBに変換
    b,g,r = cv2.split(image)       
    rgb_img = cv2.merge([r,g,b])     
    
    #tensorflow型に代入
    input_image = tf.constant(
        value=rgb_img,
        shape=(1,IMAGE_HEIGHT,IMAGE_WIDTH,CHANNEL),
        dtype=tf.float32)
    #正規化
    input_image = tf.cast(input_image, tf.float32) * (1. / 255) - 0.5
    #reshape
    input_image = tf.reshape(input_image,[1,IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    return input_image


def train(dataset):
    
    with tf.Graph().as_default(),tf.device('/gpu:0'):
        #グローバルステップ(訓練回数のカウンタ)
        global_step = tf.get_variable(
            'global_step', 
            [],
            initializer=tf.constant_initializer(0), 
            trainable=False, 
            dtype=tf.int32
        )
        num_preprocess_threads = NUM_PROCES_THREADS * NUM_GPU
        #inputのクラスからtfrecordのinputとlabel生成
        images, labels,name_string = distorted_inputs(dataset=dataset, num_preprocess_threads=num_preprocess_threads)
        input_summarys = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))
        num_classes = NUM_CLASSES
        images_splits = tf.split(axis=0, num_or_size_splits=NUM_GPU, value=images)
        labels_splits = tf.split(axis=0, num_or_size_splits=NUM_GPU, value=labels)
        #logits(出力)生成
        logits = model.vgg(image=images_splits[0],training_phase=True)
        #accuracy(正解率)
        acc= model.accuracy(logits=logits,labels=labels_splits)
        #loss(正解値からのの分散)
        losses = model.loss(logits=logits,labels=labels_splits[0])
        train_op=model.train_op(losses=losses,global_step=global_step)
        
        saver = tf.train.Saver()
        #summary...tfrecordで内容を見たい時使います.
        #checkpointの容量が大きくなる模様
        # tf.summary.scalar('loss', losses)
        # tf.summary.scalar('accuracy_rate',acc)
        # merged = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(logdir=TRAIN_DIR, graph=sess.graph)
        
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=LOG_DEVICE_PLACEMENT))
        #checkpointのカウンタ
        ckpt_point=0
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        
        #os.remove('test.txt')
        #実際のtraining部分
        for step in range(MAX_STEP):
            _,loss_value = sess.run([train_op,losses])
            if step %100 ==0:
                accuracy,lr = sess.run([acc,learning_rate])
                print("step={},accuracy = {},loss = {},lr={}".format(step,accuracy,loss_value,lr))
                #summary_writer.add_summary(summary,step)
                
            if step % 500 == 0 or (step + 1) == MAX_STEP:
                elapsed_time = time.time() - start
                f = open('time.txt', 'a')
                f.write('step={},time={}'.format(step,elapsed_time))
                f.close()

                checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                ckpt_point=step
                image_path_int+=1
                #prediction_test(ckpt = ckpt_point,i=image_path_int)
                print("add_checkpoint at {}".format(step))
                

if __name__ == '__main__':
    
    train('train.tfrecords')
    #prediction_test()
    