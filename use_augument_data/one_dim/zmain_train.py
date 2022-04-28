from use_augument_data.one_dim.cnn import train_cnn
from use_augument_data.one_dim.cnnDF import train_cnnDF
from use_augument_data.one_dim.deep_forest import train_deep_forest
from use_augument_data.one_dim.wdcnn import train_wdcnn
from use_augument_data.one_dim.wdcnnDF import train_wdcnnDF
from use_augument_data.hyper_tunnning import *
import numpy as np


def train(i=-1):
    global accuracy_list, time_list
    ## 运行在四个转速下下 （ps 这样写虽然又代码重复，但是程序不会每次都判断。就先不重构了）
    if i > 0:
        temp_list = []
        for k in range(len(function_list)):
            for n in range(len(path_list)):
                for i in range(10):
                    temp = function_list[k](path_list[n])
                    accuracy_list[path_list[n][-3:]].append(temp[0])
                    time_list[path_list[n][-3:]].append(temp[1])
            all_accuracy_list[function_list[k].__name__[6:]] = accuracy_list
            all_time_list[function_list[k].__name__[6:]] = time_list
            accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
            time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    ## 运行在某一个转速下
    else:
        for k in range(len(function_list)):
            for i in range(2):
                temp = function_list[k](path_list[0])
                accuracy_list[path_list[0][-3:]].append(temp[0])
                time_list[path_list[0][-3:]].append(temp[1])
            all_accuracy_list[function_list[k].__name__[6:]] = accuracy_list
            all_time_list[function_list[k].__name__[6:]] = time_list
            accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
            time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    write_to_file()


def write_to_file():
    with open('logs.txt', 'a') as f:  # 设置文件对象
        f.write(str(all_accuracy_list))  # 将字符串写入文件中
        f.write("\n")
        f.write(str(all_time_list))  # 将字符串写入文件中
        f.write("\n")
    for items in all_accuracy_list:
        all_accuracy_list[items] = avg(all_accuracy_list[items])
        all_time_list[items] = avg(all_time_list[items])
    with open('logs.txt', 'a') as f:  # 设置文件对象
        f.write(str(all_accuracy_list))  # 将字符串写入文件中
        f.write("\n")
        f.write(str(all_time_list))  # 将字符串写入文件中
        f.write("\n")
        f.write("######################################################################################################################################################\n")


def avg(args1):
    keys = args1.keys()
    avgs = {}
    for key in keys:
        np_item = np.array(args1[key])
        ## 有时候是百分制，有时候是
        key_avg = np.average(np_item)
        if key_avg <= 1:
            key_avg = key_avg * 100
        avgs[key] = key_avg
    print(avgs)
    return avgs

if __name__ == '__main__':
    accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    all_accuracy_list = {"cnn": accuracy_list, "cnnDF": accuracy_list, "deep_forest": accuracy_list,
                         "wdcnn": accuracy_list, "wdcnnDF": accuracy_list}
    all_time_list = {"cnn": time_list, "cnnDF": time_list, "deep_forest": time_list,
                         "wdcnn": time_list, "wdcnnDF": time_list}
    function_list = [train_cnn, train_cnnDF, train_deep_forest, train_wdcnn, train_wdcnnDF]
    train(-1)