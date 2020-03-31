import argparse
import numpy as np

parser = argparse.ArgumentParser()    # 创建解析器


# 获取参数
def get_config():
    config, unparsed = parser.parse_known_args()
    return config


# 返回bool类型的参数
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('需要bool值.')

# 数据预处理部分的参数
data_arg = parser.add_argument_group('Data')
data_arg.add_argument('--train_path', type=str, default='train_tisv', help="训练数据集路径")
data_arg.add_argument('--test_path', type=str, default='test_tisv', help="测试数据集路径")
data_arg.add_argument('--sr', type=int, default=8000, help="采样率")
data_arg.add_argument('--nfft', type=int, default=512, help="fft核尺寸")
data_arg.add_argument('--window', type=int, default=0.025, help="窗口长度 (ms)")
data_arg.add_argument('--hop', type=int, default=0.01, help="hop size (ms)")
data_arg.add_argument('--tisv_frame', type=int, default=180, help="最大帧的数目")

# LSTM模型的参数
model_arg = parser.add_argument_group('Model')
model_arg.add_argument('--hidden', type=int, default=128, help="lstm隐藏状态维数")
model_arg.add_argument('--proj', type=int, default=64, help="projection dimension of lstm")
model_arg.add_argument('--num_layer', type=int, default=3, help="lstm的层数")
model_arg.add_argument('--restore', type=str2bool, default=False, help="是否恢复模型")
model_arg.add_argument('--model_path', type=str, default='tisv_model', help="保存或加载模型的目录")
model_arg.add_argument('--model_num', type=int, default=2, help="要加载的ckpt文件的数量")

# 训练参数
train_arg = parser.add_argument_group('Training')
train_arg.add_argument('--train', type=str2bool, default=False, help="判断当前阶段，训练或推理阶段")
train_arg.add_argument('--N', type=int, default=4, help="每批次说话者的数量")
train_arg.add_argument('--M', type=int, default=5, help="每个说话者的话语数量")
train_arg.add_argument('--loss', type=str, default='softmax', help="loss类型 (softmax or contrast)")
train_arg.add_argument('--optim', type=str.lower, default='sgd', help="优化器类型")
train_arg.add_argument('--lr', type=float, default=1e-2, help="学习率")
train_arg.add_argument('--beta1', type=float, default=0.5, help="beta1")
train_arg.add_argument('--beta2', type=float, default=0.9, help="beta2")
train_arg.add_argument('--iteration', type=int, default=100000, help="最大迭代次数")
train_arg.add_argument('--comment', type=str, default='', help="any comment")

config = get_config()
print(config) # 打印所有参数