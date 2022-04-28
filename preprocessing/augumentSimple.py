#!/usr/bin/env python
# _*_ coding:utf-8 _*_
#augment
import os
from scipy.io import loadmat
import numpy as np
import random
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from collections import Counter

#这部分的思路是，通过scipy.io.loadmat 载入指定的mat文件（"FE"或者“DE”数据），
# 然后设计一个(X，y)的生成器来返回数据和数据标签；其中，X表示数据data， y是数据标签label。

def iteror_raw_data(data_path,data_mark):
    """读取.mat文件，返回数据的生成器：标签，样本数据。

    :param data_path: .mat文件所在路径
    :param data_mark: "FE",或"DE"
    :return: （标签，样本数据）
    """
    #标签数字编码 Inner Race, Outer Race, Ball
    labels = {"normal": 0, "OR021": 1, "OR014": 2, "OR007": 3, "IR021": 4, "IR014": 5, "IR007": 6, "B021": 7, "B014": 8, "B007": 9}
    # 列出所有文件
    filenams = os.listdir(data_path)
    # 逐个对mat文件进行打标签和数据提取
    filenams.sort()
    filenams.reverse()
    for single_mat in filenams:
        single_mat_path = os.path.join(data_path, single_mat)
        # 打标签
        for key, _ in labels.items():
            if key in single_mat:
                label = labels[key]

                # 数据提取
        file = loadmat(single_mat_path)
        for key, _ in file.items():
            if data_mark in key:
                data = file[key]

        yield label, data

#：（1）以时间长度s来度量数据的长度、滑动窗口的长度、重叠量的长度。目的是为了使程序适应不同采样频率的数据集。
# （2）将数据标准化设计为可选操作，这样满足对原始数据和标准化后数据的同时查看的需求，使得数据的处理更加灵活。

def data_augment(fs, win_tlen, overlap_rate, data_iteror, **kargs):
    """

    :param fs: 原始数据的采样频率
    :param win_tlen: 滑动窗口的时间长度
    :param overlap_rate: 重叠部分比例，[0-100]，百分数；
                         overlap_rate*win_tlen*fs//100 是论文中的重叠量。
    :param data_iteror: 原始数据的生成器格式
    :param kargs: {"norm"}
            norm  数据标准化的方式，三种选择：
               1："min-max";
               2:"Z-score", mean = 0,std = 1;
               3:sklearn中的StandardScaler;
    :return: (X,y):X,切分好的数据, y数据标签
    """
    overlap_rate = int(overlap_rate)
    # 重合部分的时间长度，单位s
    overlap_tlen = win_tlen * overlap_rate / 100
    # 步长，单位s
    step_tlen = win_tlen - overlap_tlen
    # 滑窗采样增强数据
    X = []
    y = []
    for iraw_data in data_iteror:
        single_raw_data = iraw_data[1]
        lab = iraw_data[0]
        number_of_win = np.floor((len(single_raw_data) - overlap_tlen * fs)
                                 / (fs * step_tlen))
        for iwin in range(1, int(number_of_win) + 1):
            # 滑窗的首尾点和其更新策略
            start_id = int((iwin - 1) * fs * step_tlen + 1)
            end_id = int(start_id + win_tlen * fs)
            current_data = single_raw_data[start_id:end_id]
            current_label = lab
            X.append(current_data)
            y.append(np.array(current_label))

    # 转换为np数组
    # X[0].shape == (win_tlen*fs, 1)
    # X.shape == (len(X), win_tlen*fs, 1)
    X = np.array(X)
    y = np.array(y)

    for key, val in kargs.items():
        # 数据标准化方式选择
        if key == "norm" and val == "1":
            X = MinMaxScaler().fit_transform(X)
        if key == "norm" and val == "2":
            X = scale(X)
        if key == "norm" and val == "3":
            X = StandardScaler().fit_transform(X)
    return X, y

#得到的数据有两个特征：（1）数据是按标签有序排列。
# （2）normal标签（0）的增强处理后数据数量是其他标签的两倍，而其他标签数据数量基本相等。
#所以该部分采取的降采样策略是，将增强处理后的normal标签（0）数据随机删除一半。

def under_sample_for_c0(X, y, low_c0, high_c0, random_seed):
    """
    使用非0类别数据的数目，来对0类别数据进行降采样。
    :param X: 增强后的振动序列
    :param y: 类别标签0-9
    :param low_c0: 第一个类别0样本的索引下标
    :param high_c0: 最后一个类别0样本的索引下标
    :param random_seed: 随机种子
    :return:
    """
    np.random.seed(random_seed)
    to_drop_ind = random.sample(range(low_c0, high_c0), (high_c0 - low_c0 + 1) - len(y[y == 3]))
    # 按照行删除
    X = np.delete(X, to_drop_ind, 0)
    y = np.delete(y, to_drop_ind, 0)
    return X, y

def preprocess(path, data_mark, fs, win_tlen,
               overlap_rate, random_seed, **kargs):
    data_iteror = iteror_raw_data(path, data_mark)
    X, y = data_augment(fs, win_tlen, overlap_rate, data_iteror, **kargs)
    # print(len(y[y==0]))
    # 降采样，随机删除类别0中一半的数据
    low_c0 = np.min(np.argwhere(y == 0))
    high_c0 = np.max(np.argwhere(y == 0))
    X, y = under_sample_for_c0(X, y, low_c0, high_c0, random_seed)
    # print(len(y[y==0]))
    print("-> 数据位置:{}".format(path))
    print("-> 原始数据采样频率:{0}Hz,\n-> 数据增强和0类数据降采样后共有：{1}条,"
          .format(fs, X.shape[0]))
    print("-> 单个数据长度：{0}采样点,\n-> 重叠量:{1}个采样点,"
          .format(X.shape[1], int(overlap_rate * win_tlen * fs // 100)))
    print("-> 类别数据数目:", (Counter(y).items()))
    return X, y

if __name__ == "__main__":
    # path = "C:\\Users\\HP5\\Desktop\\处理好的数据集\\12k_DriveEndFault\\0HP"
    path = r'../data/0HP'
    data_mark = "DE"
    fs = 12000
    win_tlen = 2048 / 12000
    overlap_rate = (2047 / 2048) * 100
    random_seed = 1
    X, y = preprocess(path,
                      data_mark,
                      fs,
                      win_tlen,
                      overlap_rate,
                      random_seed,
                      norm=3)


