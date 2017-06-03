from config import*

dataset = 'train.tfrecords'
batch_size = BATCH_SIZE
num_process_threads = None

def parse_example_proto(example_serialized):
    #入力:ファイル名
    #出力:画像,label,image_name
   
    feature_map = {
      'image_raw': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image_name': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['label'], dtype=tf.int32)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image_name = features['image_name']
    image.set_shape([IMAGE_HEIGHT * IMAGE_WIDTH * CHANNEL])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
    return image, label,image_name

def input(filenames, batch_size=BATCH_SIZE, read_threads=4, num_epochs=None):
    #filenamequeueの生成
    filename_queue = tf.train.string_input_producer([filenames], num_epochs=num_epochs, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    example_list = [parse_example_proto(serialized_example)for _ in range(read_threads)]
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch = tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch