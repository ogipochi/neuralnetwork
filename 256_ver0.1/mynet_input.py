from config import*
NUM_READER  =4
NUM_PROCES_THREADS = 4
INPUT_QUEUE_MEMORY_FACTOR = 7

def distorted_inputs(dataset, batch_size=None, num_preprocess_threads=None):
  """Generate batches of distorted versions of ImageNet images.
  Use this function as the inputs for training a network.
  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.
  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.
  Returns:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
  """
  if not batch_size:
    batch_size = BATCH_SIZE

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    images, labels,name_string = batch_inputs(
        dataset, batch_size, train=True,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=NUM_READER)
  return images, labels,name_string


def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.name_scope(values=[image_buffer], name=scope,
                     default_name='decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for name_scope.
  Returns:
    color-distorted image
  """
  with tf.name_scope(values=[image], name=scope, default_name='distort_color'):
    color_ordering = thread_id % 2

    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def distort_image(image, height, width, bbox, thread_id=0, scope=None):
  """Distort one image for training a network.
  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.
  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  with tf.name_scope(values=[image, height, width, bbox], name=scope,
                     default_name='distort_image'):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # Display the bounding box in the first thread only.
    if not thread_id:
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox)
      tf.summary.image('image_with_bounding_boxes', image_with_box)

  # A large fraction of image datasets contain a human-annotated bounding
  # box delineating the region of the image containing the object of interest.
  # We choose to create a new bounding box for the object which is a randomly
  # distorted version of the human-annotated bounding box that obeys an allowed
  # range of aspect ratios, sizes and overlap with the human-annotated
  # bounding box. If no box is supplied, then we assume the bounding box is
  # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
    if not thread_id:
      image_with_distorted_box = tf.image.draw_bounding_boxes(
          tf.expand_dims(image, 0), distort_bbox)
      tf.summary.image('images_with_distorted_bounding_box',
                       image_with_distorted_box)

    # Crop the image to the specified bounding box.
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.
    resize_method = thread_id % 4
    distorted_image = tf.image.resize_images(distorted_image, [height, width],
                                             method=resize_method)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])
    if not thread_id:
      tf.summary.image('cropped_resized_image',
                       tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors.
    distorted_image = distort_color(distorted_image, thread_id)

    if not thread_id:
      tf.summary.image('final_distorted_image',
                       tf.expand_dims(distorted_image, 0))
    return distorted_image


def eval_image(image, height, width, scope=None):
  """Prepare one image for evaluation.
  Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(values=[image, height, width], name=scope,
                     default_name='eval_image'):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=0.875)

    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    return image
def image_preprocessing(image_buffer, bbox, train, thread_id=0):
  """Decode and preprocess one image for evaluation or training.
  Args:
    image_buffer: JPEG encoded string Tensor
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    train: boolean
    thread_id: integer indicating preprocessing thread
  Returns:
    3-D float Tensor containing an appropriately scaled image
  Raises:
    ValueError: if user does not provide bounding box
  """
  if bbox is None:
    raise ValueError('Please supply a bounding box.')

  image = decode_jpeg(image_buffer)
  height = IMAGE_HEIGHT
  width = IMAGE_WIDTH

  if train:
    image = distort_image(image, height, width, bbox, thread_id)
  else:
    image = eval_image(image, height, width)

  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.
  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:
    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>
  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
  """
  # Dense features in Example proto.
  feature_map = {
      'image_raw': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image_name': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['label'], dtype=tf.int32)
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image_name = features['image_name']
  image.set_shape([IMAGE_HEIGHT * IMAGE_WIDTH * CHANNEL])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
  return image, label,image_name


def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None,
                 num_readers=1):
  """Contruct batches of training or evaluation examples from the image dataset.
  Args:
    dataset: instance of Dataset class specifying the dataset.
      See dataset.py for details.
    batch_size: integer
    train: boolean
    num_preprocess_threads: integer, total number of preprocessing threads
    num_readers: integer, number of parallel readers
  Returns:
    images: 4-D float Tensor of a batch of images
    labels: 1-D integer Tensor of [batch_size].
  Raises:
    ValueError: if data is not found
  """
  with tf.name_scope('batch_processing'):
    # data_files = dataset.data_files()
    # if data_files is None:
    #   raise ValueError('No data files found for this dataset')
    #capacityはqueueの数??
    data_files = dataset
    if train:
      filename_queue = tf.train.string_input_producer([data_files],
                                                      shuffle=True,
                                                      capacity=10000)
    else:
      filename_queue = tf.train.string_input_producer([data_files],
                                                      shuffle=False,
                                                      capacity=5000)
    if num_preprocess_threads is None:
      num_preprocess_threads =NUM_PROCES_THREADS

    if num_preprocess_threads % 4:
      raise ValueError('Please make num_preprocess_threads a multiple of 4 (%d % 4 != 0).', num_preprocess_threads)

    if num_readers is None:
      num_readers = NUM_READER

    if num_readers < 1:
      raise ValueError('Please make num_readers at least 1')

    # Approximate number of examples per shard.
    examples_per_shard = 1024
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 299*299*3*4 bytes = 1MB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 1MB = 17.6GB
    min_queue_examples = examples_per_shard * INPUT_QUEUE_MEMORY_FACTOR
    if train:
      examples_queue = tf.RandomShuffleQueue(
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples,
          dtypes=[tf.string])
    else:
      examples_queue = tf.FIFOQueue(
          capacity=examples_per_shard + 3 * batch_size,
          dtypes=[tf.string])

    # Create multiple readers to populate the queue of examples.
    if num_readers > 1:
      enqueue_ops = []
      for _ in range(num_readers):
        reader = tf.TFRecordReader()
        _, value = reader.read(filename_queue)
        enqueue_ops.append(examples_queue.enqueue([value]))

      tf.train.queue_runner.add_queue_runner(
          tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
      example_serialized = examples_queue.dequeue()
    else:
      reader = tf.TFRecordReader()
      _, example_serialized = reader.read(filename_queue)

    images_and_labels_and_name = []
    for thread_id in range(num_preprocess_threads):
      # Parse a serialized Example proto to extract the image and metadata.
      image, label_index,name_string= parse_example_proto(example_serialized)
      label_onehot = tf.one_hot(indices=label_index,depth=NUM_CLASSES,off_value=0.0,on_value=1.0)
      #image = image_preprocessing(image_buffer, bbox, train, thread_id)
      #images_and_labels.append([image, label_onehot])
      images_and_labels_and_name.append([image, label_onehot,name_string])

    images, label_index_batch ,name_string= tf.train.batch_join(
        images_and_labels_and_name,
        batch_size=batch_size,
        capacity=3 * num_preprocess_threads * batch_size)

    # Reshape images into these desired dimensions.
    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    depth = CHANNEL

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, height, width, depth])

    # Display the training images in the visualizer.
    tf.summary.image('image', images)

    label_onehot=tf.reshape(tensor=label_index_batch,shape=[BATCH_SIZE,21])
    return images,label_onehot,name_string

def inputs(dataset, batch_size=None, num_preprocess_threads=None):
  """Generate batches of ImageNet images for evaluation.
  Use this function as the inputs for evaluating a network.
  Note that some (minimal) image preprocessing occurs during evaluation
  including central cropping and resizing of the image to fit the network.
  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.
  Returns:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       image_size, 3].
    labels: 1-D integer Tensor of [FLAGS.batch_size].
  """
  if not batch_size:
    batch_size = BATCH_SIZE

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    images, labels,name_string = batch_inputs(
        dataset, batch_size, train=False,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=NUM_READER)

  return images, labels,name_string
