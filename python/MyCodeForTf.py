# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import glob
import tensorflow as tf
import sys

def tfrecordMk(dataDir1,dataDir2,record_name='train'):
    data_list=[]
    #dataDirの全ファイル名をリストで取得
    file_list = glob.glob(dataDir1+'/*')
    file_list2 = glob.glob(dataDir2+'/*')
    file_list.extend(file_list2)
    #すべてのファイルに対応するラベルをtouple型で定義
    for file in file_list:
        label = sampleLabelGetter(file)
        touple = (file,label)
        data_list.append(touple)
    
    #生成ファイル名
    record_name = record_name+'.tfrecords'

    #tfrecord生成
    writer = tf.python_io.TFRecordWriter(record_name)
    for imagePath,labels in data_list:
        print('processing {} label {}'.format(imagePath,labels))
        #画像読み込み
        load_image = Image.open(imagePath)
        #numpy型へ変換
        image = np.array(load_image)
        #バイナリ型変換
        image_name = imagePath.encode('utf-8')
        #画像サイズの読み込み
        #モデルの自動生成に必要かな
        height = image.shape[0]
        width = image.shape[1]
        #画像データは文字列にして格納する
        image_raw = image.tostring()
        #格納データの定義
        example = tf.train.Example(features=tf.train.Features(
                feature={
                    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                    'image_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_name])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels]))

                }))
        #書き込み
        writer.write(example.SerializeToString())
    writer.close()

#現段階のラベルデータつけ
#画像の名前から
def sampleLabelGetter(filename):
    folder = filename.split('/')[1]
    if folder =='background':

        return 0
    elif folder == 'character':
        return 1
    else:
        sys.exit()
    


if __name__ == '__main__':
    tfrecordMk(dataDir1='dataset/background',dataDir2='dataset/character',record_name='train')
   
