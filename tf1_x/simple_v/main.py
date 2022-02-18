import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from multiprocessing import Pool

from common_val import *
from common_utils import *
from tf_parse_tfrecord import to_tfrecord



if __name__ == "__main__":
    """
    简单版本：
    """
    raw_train_path = "raw_data"
    raw_val_path = "raw_data"
    raw_test_path = "raw_data"

    tfrecord_train_path = "tf_data"

    simple_file = os.path.join(data_path2, "sample_train1.pkl")
    # 训练文件切割
    simple_data = pd.read_pickle(simple_file)
    split_file(simple_data, 20, "./raw_data", "raw_data")

    # tfrecord生成
    n_threads = 5
    params = [[os.path.join(raw_train_path,f), 'pickle', tfrecord_train_path] for f in os.listdir(raw_train_path)]
    pool = Pool(n_threads)
    pool.map(to_tfrecord, params)
    pool.close()
    pool.join()

    # 训练模型 读取





