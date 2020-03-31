#!/usr/bin/python3.5
# -*- coding: utf-8 -*-  
 
import sys
import os
import time
import random
import numpy as np
import tensorflow as tf 
from PIL import Image
from SimpleNetwork import *
 

'''定义占位符张量，输入、输出'''
def place(SIZE,NUM_CLASSES):
    """定义placeholder节点参数"""
    # 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
    x = tf.placeholder(tf.float32, shape=[None, SIZE], name='input_x') #输入数据的大小N×D，N是数量，D是维度大小
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='output_x') #输入标签，N×C，C是类别数量
    # reshape输入图像的形状，从N，D转换为NWHC,这个数据C通道channel大小为1，N为数量
    x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1],name='in_x')
    return x,y_,x_image

"""遍历图片目录，获取图片总数，生成图片数据和标签"""
def data_preprocess(dir,NUM_CLASSES):

    count = 0 #统计图片数目
    for i in range(0,NUM_CLASSES):
        dir_2 = ''
        dir_2 = dir + '%s/' % i     # i为分类标签
        for rt, dirs, files in os.walk(dir_2):
            for filename in files:
                count += 1
 
    # 定义对应维数和各维长度的数组
    images = np.array([[0]*SIZE for i in range(count)])
    labels = np.array([[0]*NUM_CLASSES for i in range(count)])

    index = 0
    for i in range(0,NUM_CLASSES):
        dir_1 = ''
        dir_1 = dir + '%s/' % i          # 数据的目录，i为分类标签
        for rt, dirs, files in os.walk(dir_1):
            for filename in files:
                filename = dir_1 + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        images[index][w+h*width] = img.getpixel((w,h))
                labels[index][i] = 1 #将图片，即标签中每行对应的正确类别赋值为1，将标签变为onehot形式
                index += 1
    return count,images,labels

'''读取预测图片，reshape为模型输入大小，并转换为矩阵，转换到0-1'''
def img2matrix(path):
    '''
    输入：
    -path：图片路径
    输出：
    -img：0-1范围的图像矩阵1×D
    '''
    img = Image.open(path)
    img = img.resize((20,20))
    width = img.size[0]
    height = img.size[1]

    img_data = np.array([[0]*SIZE for i in range(1)])
    for h in range(0, height):
        for w in range(0, width):
            img_data[0][w+h*width] = img.getpixel((w,h))

    img = img_data.astype('float32')/255
    return img

'''预测时的函数，输入softmax概率矩阵，返回概率最大的前三个类别标签和对应的概率'''
def softmax2pre(result,NUM_CLASSES):
    max1 = 0
    max2 = 0
    max3 = 0
    max1_index = 0
    max2_index = 0
    max3_index = 0
    for j in range(NUM_CLASSES):
        if result[0][j] > max1:
            max1 = result[0][j]
            max1_index = j
            continue
        if (result[0][j]>max2) and (result[0][j]<=max1):
            max2 = result[0][j]
            max2_index = j
            continue
        if (result[0][j]>max3) and (result[0][j]<=max2):
            max3 = result[0][j]
            max3_index = j
            continue
    return max1,max2,max3,max1_index,max2_index,max3_index

'''训练模型'''
def trainModel(dir_train,dir_val,SAVER_DIR,NUM_CLASSES,iterations):
        time_begin = time.time()

        # 遍历图片目录，获取图片总数，生成图片数据和标签
        input_count,input_images,input_labels = data_preprocess(dir_train,NUM_CLASSES)
        val_count,val_images,val_labels = data_preprocess(dir_val,NUM_CLASSES)

        time_elapsed = time.time() - time_begin 
        print("读取图片文件耗费时间：%d秒" % time_elapsed)
        print ("一共读取了 %s 个训练图像， %s 个标签" % (input_count, input_count))
        print ("一共读取了 %s 个验证图像， %s 个标签" % (val_count, val_count))
        print ("train shape:", (input_images.shape))
        print ("val shape:", (val_images.shape))
        input_images.reshape((-1, 20*20))
        val_images.reshape((-1, 20*20))
        # 将数据的范围设置到0-1
        input_images = input_images.astype("float32")/255
        val_images = val_images.astype("float32")/255

        # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
        batch_size = 60
        iterations = iterations
        batches_count = int(input_count / batch_size)
        remainder = input_count % batch_size
        print ("训练数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count+1, batch_size, remainder))

        '''定义graph '''
        # 设置模型参数
        W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1), name="W_conv1")
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]), name="b_conv1")
        W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1), name="W_conv2")
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv2")
        W_fc1 = tf.Variable(tf.truncated_normal([10 * 10 * 32, 512], stddev=0.1), name="W_fc1")
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")
        W_fc2 = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev=0.1), name="W_fc2")
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b_fc2")
        keep_prob = tf.placeholder(tf.float32)
        # 计算模型，得到scores
        y_conv = model(x_image,NUM_CLASSES,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,keep_prob)
        a = tf.identity(y_conv, name='y_conv') # 命名 logits ，从而可以保存到模型中在预测时使用
        # 定义损失函数、优化器和训练op
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
        # 定义精度计算方式
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

        '''训练'''
        with tf.Session() as sess:
            time_begin = time.time()
            # 初始化网络参数
            sess.run(tf.global_variables_initializer())

            # 执行训练epoch
            for it in range(iterations):
                # 按批次训练，60大小的batch
                for n in range(batches_count):
                    train_step.run(feed_dict={x: input_images[n*batch_size:(n+1)*batch_size], y_: input_labels[n*batch_size:(n+1)*batch_size], keep_prob: 0.5})
                # 剩余的不足一个批次的数据进行训练
                if remainder > 0:
                    start_index = batches_count * batch_size
                    train_step.run(feed_dict={x: input_images[start_index:input_count-1], y_: input_labels[start_index:input_count-1], keep_prob: 0.5})

                # 每完成五次迭代，计算训练精度和验证精度 判断验证精度是否已达到100%，达到则退出迭代循环
                iterate_accuracy = 0
                if it%5 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: input_images, y_: input_labels, keep_prob: 1.0})
                    iterate_accuracy = accuracy.eval(feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})
                    print ('第 %d 次训练epoch: train准确率 %0.5f%%, val准确率 %0.5f%%' % (it, train_accuracy*100, iterate_accuracy*100))
                    if iterate_accuracy >= 0.9999 and it >= iterations:
                        break

            print ('完成训练!')
            time_elapsed = time.time() - time_begin
            print ("训练耗费时间：%d秒" % time_elapsed)
            time_begin = time.time()

            # 保存训练结果
            if not os.path.exists(SAVER_DIR):
                print ('不存在训练数据保存目录，现在创建保存目录')
                os.makedirs(SAVER_DIR)
            #  保存模型
            saver = tf.train.Saver()            
            saver_path = saver.save(sess, "%smodel.ckpt"%(SAVER_DIR))

'''使用模型进行预测'''
def testModel(SAVER_DIR,LETTERS_DIGITS,NUM_CLASSES):
        saver = tf.train.import_meta_graph("%smodel.ckpt.meta"%(SAVER_DIR))
        with tf.Session() as sess:
            model_file=tf.train.latest_checkpoint(SAVER_DIR)
            saver.restore(sess, model_file)

            # 从模型中获取训练好的参数
            W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")
            b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
            W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
            b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
            W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
            b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
            W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
            b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")
            keep_prob = tf.placeholder(tf.float32)
            # 计算模型得到scores
            y_conv = model(x_image,NUM_CLASSES,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,keep_prob)
            conv = tf.nn.softmax(y_conv)

            # 读取测试图片，并将图像转换为输入形状的矩阵
            path = u""+sys.argv[3]
            img = img2matrix(path)
            
            # 预测得到图像的类别概率矩阵
            result = sess.run(conv, feed_dict = {x: img, keep_prob: 1.0})
            
            # 得到概率最大的前三个类别标签和对应的概率
            max1,max2,max3,max1_index,max2_index,max3_index = softmax2pre(result,NUM_CLASSES)

            return max1,max2,max3,max1_index,max2_index,max3_index

if __name__ =='__main__' :

    if sys.argv[1]=='area':
        # area的参数
        SIZE = 400 #输入数据的维度大小
        WIDTH = 20 #输入数据的长宽
        HEIGHT = 20
        NUM_CLASSES = 26 #分类的类别的数量
        iterations = 500 #epoch次数
        
        SAVER_DIR = "train-saver/area/" #模型存储地址
        # AREA标签字典
        LETTERS_DIGITS = ("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z")
        # AREA训练集、验证集目录
        dir_train = './dataset/train/area/'          # 训练数据的目录
        dir_val = './dataset/val/area/'          # 验证数据的目录

        x,y_,x_image = place(SIZE,NUM_CLASSES)
        #定义训练时运行的代码 
        if sys.argv[2]=='train':
            trainModel(dir_train,dir_val,SAVER_DIR,NUM_CLASSES,iterations)
 
        #定义预测时运行的代码
        if sys.argv[2]=='predict':
            # 预测图像
            max1,max2,max3,max1_index,max2_index,max3_index = testModel(SAVER_DIR,LETTERS_DIGITS,NUM_CLASSES)
            # 得到概率最大的类别
            license_num = LETTERS_DIGITS[max1_index]

            # 输出最可能的前三个类别的概率
            print ("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (LETTERS_DIGITS[max1_index],max1*100, LETTERS_DIGITS[max2_index],max2*100, LETTERS_DIGITS[max3_index],max3*100))
            # 输出概率最大的类别
            print ("城市代号是: 【%s】" % license_num)
    
    if sys.argv[1]=='province':

        SIZE = 400
        WIDTH = 20
        HEIGHT = 20
        NUM_CLASSES = 31
        iterations = 300
        SAVER_DIR = "train-saver/province/"
        PROVINCES = ("皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新")
        dir_train = './dataset/train/province/'          # 训练数据的目录
        dir_val = './dataset/val/province/'          # 验证数据的目录

        x,y_,x_image = place(SIZE,NUM_CLASSES)
        #定义训练时运行的代码 
        if sys.argv[2]=='train':
            trainModel(dir_train,dir_val,SAVER_DIR,NUM_CLASSES,iterations)
 
        #定义预测时运行的代码 
        if sys.argv[2]=='predict':
            # 预测图像
            max1,max2,max3,max1_index,max2_index,max3_index = testModel(SAVER_DIR,PROVINCES,NUM_CLASSES)
            # 输出概率最大的类别 
            nProvinceIndex = max1_index
            print ("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (PROVINCES[max1_index],max1*100, PROVINCES[max2_index],max2*100, PROVINCES[max3_index],max3*100))
            
            print ("省份简称是: %s" % PROVINCES[nProvinceIndex])

    if sys.argv[1]=='letter':
        SIZE = 400
        WIDTH = 20
        HEIGHT = 20
        NUM_CLASSES = 34
        iterations = 300
        SAVER_DIR = "train-saver/letter/"
        LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z")
        license_num = ""
        dir_train = './dataset/train/letter/'          # 训练数据的目录
        dir_val = './dataset/val/letter/'          # 验证数据的目录
        # 定义输入、输出占位符
        x,y_,x_image = place(SIZE,NUM_CLASSES)
        #定义训练时运行的代码 
        if sys.argv[2]=='train':
            trainModel(dir_train,dir_val,SAVER_DIR,NUM_CLASSES,iterations)
 
        #定义预测时运行的代码 
        if sys.argv[2]=='predict':
            saver = tf.train.import_meta_graph("%smodel.ckpt.meta"%(SAVER_DIR))
            with tf.Session() as sess:
                model_file=tf.train.latest_checkpoint(SAVER_DIR)
                saver.restore(sess, model_file)

                # 从模型中获取训练好的参数
                W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")
                b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
                W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
                b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
                W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
                b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
                W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
                b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")
                keep_prob = tf.placeholder(tf.float32)
                # 计算模型得到scores
                y_conv = model(x_image,NUM_CLASSES,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,keep_prob)
                conv = tf.nn.softmax(y_conv)

                # 读取测试图片，并将图像转换为输入形状的矩阵(有5个数字)
                for n in range(3,8):
                    path = "./test_images/%s.bmp" % (n)
                    img = Image.open(path)
                    img = img.resize((20,20))
                    width = img.size[0]
                    height = img.size[1]

                    img_data = np.array([[0]*SIZE for i in range(1)])
                    for h in range(0, height):
                        for w in range(0, width):
                            img_data[0][w+h*width] = img.getpixel((w,h))

                    img = img_data.astype('float32')/255
                    # 预测得到图像的类别概率矩阵
                    result = sess.run(conv, feed_dict = {x: np.array(img_data), keep_prob: 1.0})
                    
                    # 得到概率最大的前三个类别标签和对应的概率
                    max1,max2,max3,max1_index,max2_index,max3_index = softmax2pre(result,NUM_CLASSES)
                    
                    license_num = license_num + LETTERS_DIGITS[max1_index]
                    print ("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (LETTERS_DIGITS[max1_index],max1*100, LETTERS_DIGITS[max2_index],max2*100, LETTERS_DIGITS[max3_index],max3*100))
                    
                print ("车牌编号是: 【%s】" % license_num)
