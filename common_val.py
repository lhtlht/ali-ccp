import pandas as pd
import numpy as np
import os
import sys
import re
import gc
import time
import warnings
warnings.filterwarnings("ignore")

root = "F:/ali-ccp"


data_path = os.path.join(root,"data")
common_features_train_csv = os.path.join(data_path, "common_features_train.csv")
common_features_test_csv = os.path.join(data_path, "common_features_test.csv")

sample_skeleton_train_csv = os.path.join(data_path, "sample_skeleton_train.csv")
sample_skeleton_test_csv = os.path.join(data_path, "sample_skeleton_test.csv")


data_path2 = data_path = os.path.join(data_path,"data_pro1")



