import tensorflow as tf
import librosa
import os
import random 
import numpy as np
import math

'''数据预处理'''
#读取音频文件，并将音频文件按照7：2：1划分为训练集，验证集，测试集；存为字典
def load_files():
    file_path = 'numbersRec/recordings/'
    file_names = os.listdir(file_path) #获取文件名称
    for i in range(0,len(file_names)):
        file_names[i] = file_path + file_names[i]
    # print(len(file_names))
    # 将不同的数字文件分到不同的数组中,并乱序
    file_name_0 = file_names[:390]
    file_name_1 = file_names[390:390*2]
    file_name_2 = file_names[390*2:390*3]
    file_name_3 = file_names[390*3:390*4]
    file_name_4 = file_names[390*4:390*5]
    file_name_5 = file_names[390*5:390*6]
    file_name_6 = file_names[390*6:390*7]
    file_name_7 = file_names[390*7:390*8]
    file_name_8 = file_names[390*8:390*9]
    file_name_9 = file_names[390*9:390*10]
    random.shuffle(file_name_0)
    random.shuffle(file_name_1)
    random.shuffle(file_name_2)
    random.shuffle(file_name_3)
    random.shuffle(file_name_4)
    random.shuffle(file_name_5)
    random.shuffle(file_name_6)
    random.shuffle(file_name_7)
    random.shuffle(file_name_8)
    random.shuffle(file_name_9)
    # print(len(file_name_0),file_name_0[-1])
    #  并分为3个数据集，然后存储到字典中
    file_name_all = [file_name_0,file_name_1,file_name_2,file_name_3,
                    file_name_4,file_name_5,file_name_6,file_name_7,
                    file_name_8,file_name_9]
    dic_train = {}
    dic_val = {}
    dic_test = {}
    for i in range(0,10):
            file_q = file_name_all[i]
            dic_train[i] = file_q[:273]
            dic_val[i] = file_q[273:273+78]
            dic_test[i] = file_q[273+78:-1]
    # print(dic_train[0])
    return dic_train,dic_val,dic_test


# 将 标签转换为one hot向量
def dense_to_one_hot(label,num):
    '''
    label:真实的标签类别
    num:总的类别数
    '''
    a = [0]*num
    a[label] = 1
    # print(a)
    return a
# 对音频文件特征MFCC进行提取
def read_files(files):
    labels = []
    features = []
    for ans, files in files.items():
        for file in files:
            wave, sr = librosa.load(file, mono=True)
            label = dense_to_one_hot(ans, 10)
            labels.append(label)
            mfcc = librosa.feature.mfcc(wave, sr)
            mfcc = np.pad(mfcc, ((0, 0), (0, 100 - len(mfcc[0]))), mode='constant', constant_values=0)
            features.append(np.array(mfcc))
    return np.array(features), np.array(labels)
# 对特征向量进行归一化处理
def mean_normalize(features):
    std_value = features.std()
    mean_value = features.mean()
    return (features - mean_value) / std_value

# 神经网络配置文件
class CNNConfig(object):
    def __init__(self):
        self.filter_sizes = [2,3,4,5]
        self.num_filters = 64
        self.hidden_dim = 1000
        self.learning_rate = 0.0001
        self.num_epochs = 500
        self.dropout_keep_prob = 0.5
        self.print_per_batch = 100
        self.save_tb_per_batch = 100

'''构建模型：ASRCNN'''
class ASRCNN(object):
    def __init__(self, config, width, height, num_classes):  # 20,100
        self.config = config
        self.input_x = tf.placeholder(tf.float32, [None, width, height], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # input_x = tf.reshape(self.input_x, [-1, height, width])
        input_x = tf.transpose(self.input_x, [0, 2, 1])
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size)
                conv = tf.layers.conv1d(input_x, self.config.num_filters, filter_size, activation=tf.nn.relu)
                pooled = tf.reduce_max(conv, reduction_indices=[1])
                pooled_outputs.append(pooled)
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)  # 64*4
        pooled_reshape = tf.reshape(tf.concat(pooled_outputs, 1), [-1, num_filters_total])
        #pooled_flat = tf.nn.dropout(pooled_reshape, self.keep_prob)

        fc = tf.layers.dense(pooled_reshape, self.config.hidden_dim, activation=tf.nn.relu, name='fc1')
        fc = tf.contrib.layers.dropout(fc, self.keep_prob)
        #fc = tf.nn.relu(fc)
        # 分类器
        self.logits = tf.layers.dense(fc, num_classes, name='fc2')
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")  # 预测类别
        # 损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy)
        # 优化器
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        # 准确率
        correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 将训练数据分为按批次迭代的数据
def batch_iter(train_features, train_labels):
    # print(train_features.shape)
    X = train_features
    Y = train_labels
    mini_batch_size = 64

    np.random.seed(0) #指定随机种子
    m = X.shape[0]
    mini_batches = []

    #第一步：打乱顺序
    permutation = list(np.random.permutation(m)) #它会返回一个长度为m的随机数组，且里面的数是0到m-1
    shuffled_X = X[permutation]  #将每一列的数据按permutation的顺序来重新排列。
    shuffled_Y = Y[permutation]

    #第二步，分割
    num_complete_minibatches = math.floor(m / mini_batch_size) 
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    #如果训练集的大小刚好是mini_batch_size的整数倍，那么这里已经处理完了
    #如果训练集的大小不是mini_batch_size的整数倍，那么最后肯定会剩下一些，要把它处理了
    if m % mini_batch_size != 0:
        #获取最后剩余的部分
        mini_batch_X = shuffled_X[mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[mini_batch_size * num_complete_minibatches:]

        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

'''模型训练'''
def train(argv=None):
    '''batch = mfcc_batch_generator()
    X, Y = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y  # overfit for now'''
    train_files, valid_files, test_files = load_files()
    train_features, train_labels = read_files(train_files)
    train_features = mean_normalize(train_features)
    print('read train files down')
    valid_features, valid_labels = read_files(valid_files)
    valid_features = mean_normalize(valid_features)
    print('read valid files down')
    test_features, test_labels = read_files(test_files)
    test_features = mean_normalize(test_features)
    print('read test files down')

    width = 20  # mfcc features
    height = 100  # (max) length of utterance
    classes = 10  # digits

    config = CNNConfig()
    cnn = ASRCNN(config, width, height, classes)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join('cnn_model', 'model.ckpt')
    tensorboard_train_dir = 'tensorboard/train'
    tensorboard_valid_dir = 'tensorboard/valid'

    if not os.path.exists(tensorboard_train_dir):
        os.makedirs(tensorboard_train_dir)
    if not os.path.exists(tensorboard_valid_dir):
        os.makedirs(tensorboard_valid_dir)
    tf.summary.scalar("loss", cnn.loss)
    tf.summary.scalar("accuracy", cnn.acc)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    valid_writer = tf.summary.FileWriter(tensorboard_valid_dir)

    total_batch = 0
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(train_features, train_labels)
        for x_batch, y_batch in batch_train:
            total_batch += 1
            # feed_dict = feed_data(cnn, x_batch, y_batch, config.dropout_keep_prob)
            session.run(cnn.optim, feed_dict={cnn.input_x: x_batch,
                                                cnn.input_y: y_batch,
                                                cnn.keep_prob: config.dropout_keep_prob})
            if total_batch % config.print_per_batch == 0:
                train_loss, train_accuracy = session.run([cnn.loss, cnn.acc], feed_dict={cnn.input_x: x_batch,
                                                                                         cnn.input_y: y_batch,
                                                                                         cnn.keep_prob: config.dropout_keep_prob})
                valid_loss, valid_accuracy = session.run([cnn.loss, cnn.acc], feed_dict={cnn.input_x: valid_features,
                                                                                         cnn.input_y: valid_labels,
                                                                                         cnn.keep_prob: config.dropout_keep_prob})
                print('Steps:' + str(total_batch))
                print(
                    'train_loss:' + str(train_loss) + ' train accuracy:' + str(train_accuracy) + '\tvalid_loss:' + str(
                        valid_loss) + ' valid accuracy:' + str(valid_accuracy))
            if total_batch % config.save_tb_per_batch == 0:
                train_s = session.run(merged_summary, feed_dict={cnn.input_x: x_batch,
                                                                    cnn.input_y: y_batch,
                                                                    cnn.keep_prob: config.dropout_keep_prob})
                train_writer.add_summary(train_s, total_batch)
                valid_s = session.run(merged_summary, feed_dict={cnn.input_x: valid_features, cnn.input_y: valid_labels,
                                                                 cnn.keep_prob: config.dropout_keep_prob})
                valid_writer.add_summary(valid_s, total_batch)

        saver.save(session, checkpoint_path, global_step=epoch)
    test_loss, test_accuracy = session.run([cnn.loss, cnn.acc],
                                           feed_dict={cnn.input_x: test_features, cnn.input_y: test_labels,
                                                      cnn.keep_prob: config.dropout_keep_prob})
    print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))

'''模型使用'''
# 加载模型，使用测试数据，输出音频识别结果
# 测试数据准备,读取文件并提取音频特征
def read_test_wave(path):
    files = os.listdir(path)
    feature = []
    features = []
    label = []
    for wav in files:
        # print(wav)
        if not wav.endswith(".wav"): continue
        ans = int(wav[0])        
        wave, sr = librosa.load(path+wav, mono=True)
        label.append(ans)
        # print("真实lable: %d" % ans)
        mfcc = librosa.feature.mfcc(wave, sr)
        mfcc = np.pad(mfcc, ((0, 0), (0, 100 - len(mfcc[0]))), mode='constant', constant_values=0)
        feature.append(np.array(mfcc))   
    features = mean_normalize(np.array(feature))
    return features,label

# 模型加载
def test(path):
    features, label = read_test_wave(path)
    print('loading ASRCNN model...')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('cnn_model/model.ckpt-499.meta')
        saver.restore(sess, tf.train.latest_checkpoint('cnn_model'))  
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_x:0")
        pred = graph.get_tensor_by_name("pred:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        for i in range(0, len(label)):
            feed_dict = {input_x: features[i].reshape(1,20,100), keep_prob: 1.0}
            test_output = sess.run(pred, feed_dict=feed_dict)
            
            print("="*15)
            print("真实lable: %d" % label[i])
            print("识别结果为:"+str(test_output[0]))
        print("Congratulation!")  

if __name__ == "__main__":
    # load_files()
    train()