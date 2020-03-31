import tensorflow as tf
import os
from model import train, test
from config import get_config

config = get_config()
tf.reset_default_graph()

if __name__ == "__main__":
    # 训练
    if config.train:
        os.makedirs(config.model_path)
        train(config.model_path)
    # 测试
    else:
        test(config.model_path)
