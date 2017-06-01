from config import*

#############################################################################################
#モデルに使用する層を定義

#畳み込み層
def conv_layer(
    bottom,
    name,
    num_filters,
    training_phase=True,
    strides = (1,1),
    dilation_rate=(1,1),
    kernel_size=[2,2]):

    #bottom:入力
    #name:層のまとまりの名前
    #num_filters:フイルターの名前
    #training_phase:訓練か否か
    #stride:畳み込み層のstride
    #dilation_rate:畳み込みの値のとり方の隙間のとりぐあい
    #               defalutの(1,1)は隙間なしを意味する
    #kernel_size:カーネルサイズ

    with tf.variable_scope(name):
        first = tf.layers.conv2d(
            inputs=bottom,
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            trainable=training_phase,
            strides=strides,
            padding='SAME', name='conv1')
        second = tf.contrib.layers.batch_norm(inputs=first, center=True, scale=True, is_training=training_phase, scope='bn')
        relu = tf.nn.relu(second, 'relu')
        return relu
#プーリング層
def max_pool(
    bottom,
    name,
    pool_size=[2,2],
    strides=(1,1)):
    #bottom:入力
    #name:層のまとまりの名前
    #pool_size:フィルタサイズ
    #strides:stride
    return tf.layers.max_pooling2d(inputs=bottom,pool_size=pool_size,strides=strides,padding='SAME',name=name)


def fc_layer(
    bottom,
    name,
    units,
    training_phase=True):
    #bottom:入力
    #name:層のまとまりの名前
    #units:出力フィルタ数
    #training_phase:値の訓練を行うか否か
    with tf.variable_scope(name):
        first = tf.layers.dense(inputs=bottom,units=units,name='fc1',trainable=training_phase)
        second = tf.contrib.layers.batch_norm(inputs=first,center=True,scale=True,is_training=training_phase)
        relu = tf.nn.relu(second,'relu')
        return relu
################################################################################################
#モデルの定義
def vgg(image,training_phase):
    conv1_1=conv_layer(bottom=image,name='conv1_1',num_filters=32,kernel_size=[3,3],strides = (2,2),training_phase=training_phase)
    pool1 = max_pool(bottom=conv1_1,name='pool1',pool_size=[2,2])
    conv2_1 = conv_layer(bottom=pool1,name='conv2_1',num_filters=64,kernel_size=[3,3],strides = (2,2),training_phase=training_phase)
    conv2_2 = conv_layer(bottom=conv2_1,name='conv2_2',num_filters=64,kernel_size=[3,3],strides = (2,2),training_phase=training_phase)
    pool2 = max_pool(bottom=conv2_2,name='pool2')
    conv3_1 = conv_layer(bottom=pool2,name='conv3_1',num_filters=128,kernel_size=[3,3],strides = (2,2),training_phase=training_phase)
    conv3_2=conv_layer(bottom=conv3_1,name='conv3_2',num_filters=128,kernel_size=[3,3],strides = (2,2),training_phase=training_phase)
    pool3 = max_pool(bottom=conv3_2,name='pool3')
    conv4_1 = conv_layer(bottom=pool3,name='conv4_1',num_filters=256,kernel_size=[3,3],strides = (2,2),training_phase=training_phase)
    conv4_2 = conv_layer(bottom=conv4_1,name='conv4_2',num_filters=256,kernel_size=[3,3],strides = (2,2),training_phase=training_phase)
    pool4 = max_pool(bottom=conv4_2,name='pool4')
    flatten = slim.layers.flatten(inputs=pool4,scope='flatten')
    fc1=fc_layer(bottom=flatten,name='fc1',units=256,training_phase=training_phase)
    fc2 = fc_layer(bottom=fc1, name='fc2', units=256,training_phase=training_phase)
    fc3 = fc_layer(bottom=fc2,name='fc3',units=NUM_CLASSES,training_phase=training_phase)
    return  fc3
##################################################################################################
#lossの定義
def loss(logits,labels):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits)
    tf.summary.scalar("loss",loss)
    return loss

###############################################################################################
#accuracyの定義
def accuracy(logits,labels):
    correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    tf.summary.scalar("accuracy",accuracy)
    return accuracy

##############################################################################################
#train_opの定義
def train_op(
    losses,
    global_step,
    initial_learning_rate=INITIAL_LEARNING_RATE,
    batch_size=BATCH_SIZE
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH,
    learning_rate_decay_factor = LEARNING_RATE_DECAY_FACTOR,
    num_epochs_per_decay = NUM_EPOCHS_PER_DECAY,
    rmsprop_decay = RMSPROP_DECAY,
    RMSPROP_MOMENTUM = rmsprop_momentam,
    rmsprop_epsilon = RMSPROP_EPSILON
    ):



    num_batches_per_epoch = num_examples_per_epoch / batch_size
    decay_step = num_batches_per_epoch * num_epochs_per_decay
    learning_rate = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=learning_rate_decay_factor,
        staircase=True)

    # optimizer = tf.train.RMSPropOptimizer(
    #     learning_rate=learning_rate, 
    #     decay=RMSPROP_DECAY,
    #     momentum=RMSPROP_MOMENTUM,
    #     epsilon=RMSPROP_EPSILON)

    #新しいoptimizerあったので変えてみた
    #そのパラメータはデフォルト値が論文推奨の値になっている
    optimizer =tf.train.AdamOptimizer(learning_rate=learning_rate) 

    train_op = tf.contrib.layers.optimize_loss(
            loss=losses,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=optimizer)
            
    return train_op