import numpy as np
from PIL import Image
import glob
import tensorflow as tf

def　tfrecordMk(dataDir,record_name='train'):
    data_list=[]
    #dataDirの全ファイル名をリストで取得
    file_list = glob.glob(dataDir+'/*')
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
        print('processing 'imagePath)
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
    if '0001' in filename:
        return 1
    elif '0002' in filename:
        return 2
    elif '0003' in filename:
        return 3
    elif '0004' in filename:
        return 4
    elif '0005' in filename:
        return 5
    elif '0006' in filename:
        return 6
    elif '0007' in filename:
        return 7
    elif '0008' in filename:
        return 8
    elif '0009' in filename:
        return 9
    elif '0010' in filename:
        return 10
    elif '0011' in filename:
        return 11
    elif '0012' in filename:
        return 12
    elif '0013' in filename:
        return 13
    elif '0014' in filename:
        return 14
    elif '0015' in filename:
        return 15
    elif '0016' in filename:
        return 16
    elif '0017' in filename:
        return 17
    elif '0018' in filename:
        return 18
    elif '0019' in filename:
        return 19
    elif '0020' in filename:
        return 20
    elif '0021' in filename:
        return 0
    


if __name__ == '__main__':
    tfrecordMk(dataDir='256x256',record_name='train')
   
