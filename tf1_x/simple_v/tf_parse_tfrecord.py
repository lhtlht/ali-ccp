import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

def parse_line(line):
    pass

def parse_line_row(line):
    features = dict()
    # feature
    gender = [str(line.gender).encode('utf-8')]
    features['gender'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=gender))
    features['age'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[line.age]))
    features['itemid'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[line.itemid]))

    user_intentions = ['0'.encode('utf-8')]*(10-len(line.user_intentions)) + [i.encode('utf-8') for i in line.user_intentions]
    features['user_intentions'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=user_intentions))

    user_cate_val = str(line.user_cate_val) if str(line.user_cate_val) != 'nan' else '0'
    user_shop_val = str(line.user_shop_val) if str(line.user_shop_val) != 'nan' else '0'
    user_brand_val = str(line.user_brand_val) if str(line.user_brand_val) != 'nan' else '0'
    combine_val = np.array([user_cate_val, user_shop_val, user_brand_val], dtype=np.float64)
    features['combine_val'] = tf.train.Feature(float_list=tf.train.FloatList(value=combine_val))

    # label
    features['click'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[line.click]))
    features['conversion'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[line.conversion]))

def to_tfrecord(params):
    assert len(params) <= 3, f"params must contains input file and output path, but {params}"
    in_file = params[0]
    in_type = params[1]
    out_dir = params[2]
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    basename = os.path.basename(in_file).split('.')[0]
    output_tfrecord = tf.io.TFRecordWriter(f"{out_dir}/{basename}.tfrecord")
    if in_type == "txt":
        with open(in_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                features = parse_line(line)
                examples = tf.train.Example(features=tf.train.Features(feature=features))
                serialized = examples.SerializeToString()
                output_tfrecord.write(serialized)
    elif in_type == "pickle":
        data = pd.read_pickle(in_file)
        for line in data.itertuples():
            features = parse_line_row(line)
            examples = tf.train.Example(features=tf.train.Features(feature=features))
            serialized = examples.SerializeToString()
            output_tfrecord.write(serialized)
    output_tfrecord.close()