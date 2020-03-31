#encoding=utf8
import tensorflow as tf
    
# 定义卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')
 

# 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)

def model(x_image,NUM_CLASSES,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,keep_prob):           

    # 第一个卷积层+ReLU+最大池化
    conv_strides = [1, 1, 1, 1]
    kernel_size = [1, 2, 2, 1]
    pool_strides = [1, 2, 2, 1]
    L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

    # 第二个卷积层+ReLU+最大池化
    conv_strides = [1, 1, 1, 1]
    kernel_size = [1, 1, 1, 1]
    pool_strides = [1, 1, 1, 1]
    L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')


    # 全连接层1+ReLU
    h_pool2_flat = tf.reshape(L2_pool, [-1, 10 * 10 * 32]) # 将输出数据展开，N×D
    h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

    # dropout

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 全连接层2
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # 返回计算得到的logits
    return y_conv