import numpy as np
import tensorflow as tf
import os
import sys
import collections
import time

# 存放数据的路径 
DATA_PATH = "PTB_dataset"
# 参数 
hidden_size = 200 # 单词向量的维度，隐藏层，用于记忆和储存过去状态的节点个数 
num_layers = 3 # LSTM结构的层数为2层，前一层的LSTM的输出作为后一层的输入 
vocab_size = 10000 # 词典大小，可以存储10000个 
learning_rate = 1.0 # 初始学习率 
train_batch_size = 60 # 训练batch大小 
train_num_step = 40 # 一个训练序列长度 
num_epoch = 20
keep_prob = 0.5 # 节点保存50% 
max_grad_norm = 5 # 用于控制梯度膨胀（误差对输入层的偏导趋于无穷大） 

# 在验证和测试时不用限制序列长度 ,
eval_batch_size = 1 
eval_num_step = 1 

'''读取数据，并分割为数组，元素为单词string'''
def _read_words(filename):
    '''从文件中读取数据并分割'''
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

'''将单词按照出现的次数降序排序，返回字典
key为单词string
value为id （id从0到len-1）
'''
def _build_vocab(filename):
    # 将单词文件转换为数组
    data = _read_words(filename)
    # 统计每个单词出现的次数
    counter = collections.Counter(data)
    # 返回排序数组，包括单词和次数，首先按照单词出现的次数降序排序，次数相同时，按照单词升序排序
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # 返回出现次数从大到小的单词组成的列表，没有次数。单词顺序确定
    words, _ = list(zip(*count_pairs))
    # 返回字典，key为单词，value为对应的id
    word_to_id = dict(zip(words, range(len(words))))
    '''
    该函数的效果如下
    counter Counter({'cc': 3, 'aaa': 2, 'bbb': 2, 'b': 1})
    counter.items() dict_items([('aaa', 2), ('bbb', 2), ('b', 1), ('cc', 3)])
    count_pairs [('cc', 3), ('aaa', 2), ('bbb', 2), ('b', 1)]
    words ('cc', 'aaa', 'bbb', 'b')
    word_to_id {'cc': 0, 'aaa': 1, 'bbb': 2, 'b': 3}
    '''
    return word_to_id

'''
返回单词在字典中对应的id组成的数组
'''
def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

'''读取数据，返回训练集、测试集、验证集
返回三个数据集对应的三个数组，每个数组由由每个单词对应的id组成
'''
def ptb_raw_data(data_path=None):
    """从"data_path"加载PTB行数据.
    读取PTB text files, 将字符串转换为id,
    Args:
    data_path: 数据路径
    Returns:
    张量 (train_data, valid_data, test_data, vocabulary)
    """
    # 获取三个数据集的路径
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    
    # 生成字典：根据训练集中的单词返回字典，key：单词，value：id
    word_to_id = _build_vocab(train_path)
    # 返回三个数据集。数组，由每个单词对应的id组成
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary

'''数据处理函数'''
#从文件读入的语料库数据上迭代生成训练、验证、训练数据集
#其中raw_data是需要将词转换成词序号，方便后续处理
def ptb_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32) # 将数据转换为张量
        #将数据进行切分，去掉多余数据
        data_len = tf.size(raw_data) #数据的总量
        batch_len = data_len // batch_size #每个批次中的数据量
        #将数据截断为整数批次，并去掉多余的数据
        # batch_size,一个批次的大小
        # batch_len,批次的数量
        data = tf.reshape(raw_data[0 : batch_size * batch_len], 
                        [batch_size, batch_len])
        # 每个批次中的序列数量
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        #从数据中获得批数据集
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                            [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                            [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y

'''模型构建'''
class PTBModel(object): # 类要使用camelcase格式 
    def __init__(self, is_training, batch_size, num_steps): # 初始化属性 
        self.batch_size = batch_size 
        self.num_steps = num_steps 
        
        # 定义输入层，输入层维度为batch_size * num_steps 
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps]) 
        # 定义正确输出 
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps]) 
        
        # 定义lstm结构 
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size) 
        if is_training: 
            # 使用dropout 
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob) 
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers) # 实现多层LSTM 
        
        # 将lstm中的状态初始化为全0数组，BasicLSTMCell提供了zero_state来生成全0数组 
        # batch_size给出了一个batch的大小 
        self.initial_state = cell.zero_state(batch_size, tf.float32) 
        # 生成单词向量，单词总数为10000，单词向量维度为hidden_size200，所以词嵌入总数embedding为 
        embedding = tf.get_variable("embedding", [vocab_size, hidden_size]) 
        
        # lstm输入单词为batch_size*num_steps个单词，则输入维度为batch_size*num_steps*hidden_size 
        # embedding_lookup为将input_data作为索引来搜索embedding中内容，若input_data为[0,0],则输出为embedding中第0个词向量 
        inputs = tf.nn.embedding_lookup(embedding, self.input_data) 
        
        # 在训练时用dropout 
        if is_training: 
            inputs = tf.nn.dropout(inputs, keep_prob) 
            
        # 输出层 
        outputs = [] 
        # state为不同batch中的LSTM状态，初始状态为0 
        state = self.initial_state 
        with tf.variable_scope("RNN"): 
            for time_step in range(num_steps): 
                if time_step > 0: 
                    # variables复用 
                    tf.get_variable_scope().reuse_variables() 
                # 将当前输入进lstm中,inputs输入维度为batch_size*num_steps*hidden_size 
                cell_output, state = cell(inputs[:, time_step, :], state) 
                # 输出队列 
                outputs.append(cell_output) 
        
        # 输出队列为[batch, hidden_size*num_steps]，在改成[batch*num_steps, hidden_size] 
        # [-1, hidden_size]中-1表示任意数量的样本 
        output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size]) 
        
        # lstm的输出经过全连接层得到最后结果，最后结果的维度是10000，softmax后表明下一个单词的位置（概率大小） 
        weight = tf.get_variable("weight", [hidden_size, vocab_size]) 
        bias = tf.get_variable("bias", [vocab_size]) 
        logits = tf.matmul(output, weight) + bias # 预测的结果 
        
        # 交叉熵损失，tensorflow中有sequence_loss_by_example来计算一个序列的交叉熵损失和 
        # tf.reshape将正确结果转换为一维的,tf.ones建立损失权重，所有权重都为1，不同时刻不同batch的权重是一样的 
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.targets, [-1])], 
            [tf.ones([batch_size * num_steps], dtype=tf.float32)]) 
        
        # 每个batch的平均损失,reduce_sum计算loss总和 
        self.cost = tf.reduce_sum(loss)/batch_size 
        self.final_state = state 
        
        # 在训练时定义反向传播 
        if not is_training: 
            return 
        self._lr = tf.Variable(0.0, trainable=False)
        trainable_variables = tf.trainable_variables() 
        # 使用clip_by_global_norm控制梯度大小，避免梯度膨胀 
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), max_grad_norm) 
        # 梯度下降优化 
        optimizer = tf.train.GradientDescentOptimizer(self._lr) 
        # 训练步骤,apply_gradients将计算出的梯度应用到变量上 
        # zip将grads和trainable_variables中每一个打包成元组 
        # a = [1,2,3]， b = [4,5,6]， zip(a, b)： [(1, 4), (2, 5), (3, 6)] 
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables)) 

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
        
'''在单个epoch中对mini batch进行迭代'''
def run_epoch(session, model, data, train_op, output_log, epoch_size): 
    # perplexity是用来评价一个语言模型预测一个样本是否很好的标准。越低，代表模型的预测性能越好 
    start_time = time.time()
    total_costs = 0.0 
    iters = 0 
    state = session.run(model.initial_state) 
    
    # 训练一个epoch 
    for step in range(epoch_size): 
        x, y = session.run(data) 
        # cost是交叉熵损失，即下一个单词为给定单词的概率 
        cost, state, _ = session.run([model.cost, model.final_state, train_op], 
            {model.input_data: x, model.targets: y, model.initial_state: state}) 
        # 将所有batch、时刻的损失相加 
        total_costs += cost 
        # 所有epoch总输出单词数 
        iters += model.num_steps 
        
        if output_log and step % 100 == 0: 
            print(" %d steps, train_perplexity: %.3f, speed: %.2f wps" % 
            (step, np.exp(total_costs / iters),
            iters * model.batch_size  /(time.time() - start_time)))
        
        
            
    # 返回语言模型的perplexity值 
    return np.exp(total_costs / iters) 
    
def main(): 
    
    # 获取数据，将数据分为训练集、验证集、测试集 
    train_data, valid_data, test_data, _ = ptb_raw_data(DATA_PATH) 
    
    # 计算单个epoch中需要训练batch的次数 
    train_epoch_size = ((len(train_data) // train_batch_size) - 1) // train_num_step
    valid_epoch_size = ((len(valid_data) // train_batch_size) - 1) // train_num_step
    test_epoch_size = ((len(test_data) // train_batch_size) - 1) // train_num_step
    
    # 定义初始化函数 ，初始化权重和偏置
    initializer = tf.random_uniform_initializer(-0.05, 0.05) 
    
    # 定义语言训练模型 
    with tf.name_scope("Train"):
        with tf.variable_scope("language_model", reuse=None, initializer=initializer): 
            train_model = PTBModel(True, train_batch_size, train_num_step) 
            # tf.summary.scalar("training loss",train_model.cost)
            # tf.summary.scalar("Learning Rate", train_model._lr)

       
    # 定义语言测试模型 ，测试和验证
    with tf.name_scope("eval"):
        with tf.variable_scope("language_model", reuse=True, initializer=initializer): 
            eval_model = PTBModel(False, train_batch_size, train_num_step) 
            # tf.summary.scalar("validation loss",eval_model.cost)

    
    # 训练模型 
    with tf.Session() as session: 
        # 初始化
        tf.global_variables_initializer().run() 

        '''用作tensorboard的可视化'''
        # #将可视化的参数进行综合，merged
        # merged = tf.summary.merge_all()
        
        # ##设置日志记录的路径
        # writer = tf.summary.FileWriter("logs/", session.graph)

        #将训练集、测试集、验证集转换为迭代的小批次数据
        train_queue = ptb_producer(train_data, train_model.batch_size, train_model.num_steps) 
        eval_queue = ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps) 
        test_queue = ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps) 
        
        # 并行化处理
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=session, coord=coord) 
        
        # 迭代num_epoch次数据
        for i in range(num_epoch): 
            print("iteration: %d" % (i + 1)) 
            # 更新学习率
            lr_decay = 0.95 ** max(i - 6, 0.0)
            # lr_decay = 0.9**i
            train_model.assign_lr(session, learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(train_model._lr)))

            # 按批次进行训练，训练一个epoch
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size) 
            # 传入了tf.no_op表示不进行优化 ，输出整个验证集的perplexity值
            valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size) 
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity)) 
            # 记录日志，每隔一个epoch记录一次数据
            # rs=session.run(merged)
            # writer.add_summary(rs, i)
        #训练完成之后，在测试集上运行网络
        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size) 
        print("Test Perplexity: %.3f" % test_perplexity) 
        
        coord.request_stop() 
        coord.join(threads) 

        # 保存模型
        save_path = "result/model"
        print("Saving model to %s." % save_path)
        saver = tf.train.Saver()
        save_path = saver.save(session, save_path)


if __name__ == "__main__": 
    main()
