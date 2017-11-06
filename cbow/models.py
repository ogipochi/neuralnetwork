import tensorflow as tf
import settings
import numpy as np
import text_helpers
import os

class CBOW(object):
    def __init__(
        self,
        batch_size,
        embedding_size,
        vocabrary_size,
        generations,
        model_learning_rate,
        num_sampled,
        window_size,
        save_embedding_every,
        print_valid_every,
        print_loss_every):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.vocabrary_size = vocabrary_size
        self.generations = generations
        self.model_learning_rate = model_learning_rate
        self.num_sampled = num_sampled
        self.window_size = window_size
        self.save_embedding_every = save_embedding_every
        self.print_valid_every = print_valid_every
        self.print_loss_every = print_loss_every
    def inference(self,x_inputs):
        embed = tf.zeros(
            shape=[self.batch_size,self.embedding_size]
            )
        
        for element in range(2*self.window_size):
            embed +=tf.nn.embedding_lookup(
                embeddings,
                x_inputs[:,element]
            )
        return embed
    def loss(self,labels,y_inputs):
        #NCE損失関数パラメータ
        nce_weights = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[self.vocabrary_size,self.embedding_size],
                stddev=1.0/np.sqrt(self.embedding_size)
                ))
        nce_biases = tf.Variable(
            tf.zeros(shape=[self.vocabrary_size])
        )
        loss_value = tf.reduce_mean(tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels = labels,
            inputs= y_inputs,
            num_sampled=self.num_sampled,
            num_classes=self.num_classes
            ))
    
    
    def train(self,loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.model_learning_rate)
        train_step = optimizer.minimize(loss)
        return train_step
    
    def accuracy(self,embeggings,):
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        return similarity
    def fit(self,valid_examples,text_data,valid_words,word_dict,word_dict_rev):
        sess = tf.Session()
        #単語埋め込みを定義
        embeddings=tf.Variable(tf.random_uniform(shape=[self.vocabrary_size,self.embedding_size],-1.0,1.0))
        #プレースホルダを作成
        #入力,対象の単語の上下window_size分切り取ったbatchsizeの数の行列
        x_inputs = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,2 * self.window_size])
        #出力,目的の単語のindex
        y_target = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,1])
        #テストデータセット
        valid_dataset = tf.constant(valid_examples,dtype=tf.int32)
        #モデル
        embed = self.inference(x_inputs=x_inputs,)
        #損失関数
        loss = self.loss(labels=y_target,y_inputs=embed)

        similarity = self.accuracy(embeddings=embeddings)

        #埋め込み保存用Saver定義
        #デフォルトではモデル全体が保存されるが,
        #今回の記述方法では埋め込みだけが保存されるので注意
        saver = tf.train.Saver(var_list=None{"embeddings":embeddings})

        #train_step
        train_step = self.train(loss=loss)
        #初期化関数
        init = tf.global_variables_initializer()
        sess.run(init)
        #テキストデータは最低設定したwindow size ×2 + 1の単語数がないと行けないため,条件を満たさないデータを削除する
        text_data = [x for x in text_data if len(x) >= (2*self.window_size+1)]

        ##############################################
        #トレーニングループ
        ##############################################
        loss_vec = []
        loss_x_vec = []

        for i in range(self.generations):
            #自作のモジュールtext_helpersのメソッドgenerate_batch_dataを使い,
            batch_inputs ,batch_labels = text_helpers.generate_batch_data(
                sentences=text_data,
                batch_size=batch_size,
                window_size=window_size,
                method='cbow',
            )
            feed_dict = {x_inputs:batch_inputs,y_target:batch_labels}
            #trainingの実行
            _train_step, = sess.run(train_step)

            #print_loss_everyごとにlossを計算,グラフ表示用のベクトルに値を追加し,その値を表示
            if (i + 1) % self.print_loss_every == 0:
                loss_val = sess.run(loss,feed_dict=feed_dict)
                loss_vec.append(loss_val)
                loss_x_vec.append(i+1)
                print('{}:Loss={}'.format(i+1,loss_val))
            
            if (i + 1) % self.print_valid_every == 0:
                sim = sess.run(similarity,feed_dict=feed_dict)
                for j in range(len(valid_words)):
                    #テストワードの値からベクトルを取得
                    valid_words = word_dict_rev[valid_examples[j]]
                    top_k = 5
                    nearest = (-sim[j,:]).argsort()[a:top_k+1]
                    log_str = "nearest to {}".format(valid_words)
                    for k in range(top_k):
                        close_word = word_dict_rev[nearest[k]]
                        print_str = '{}{}'.format(log_str,close_word)
                        print(print_str)
                
            if(i+1) % self.save_embedding_every == 0:
                #語彙ディクショナリを保存
                with open(os.path.join('movie_vocab.pkl'),'wb') as f:
                    pickle.dump(word_dict,f)

                model_checkpoint_path = ('cbow_movie_enbedding.ckpt')
                save_path = saver.save(sess,model_checkpoint_path)
                print('model saved in file:{}'.format(save_path))
                
