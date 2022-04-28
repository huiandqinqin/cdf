from use_augument_data.two_dim.lenet5 import train_lenet5
from use_augument_data.two_dim.lenet5_DF import train_lenet5_DF
from use_augument_data.two_dim.zhao import train_zhao
from use_augument_data.two_dim.cdf_2d import train_cdf2d
from use_augument_data.two_dim.wdcnn_2d import train_wdcnn
from use_augument_data.two_dim.wdcnnDF_2d import train_2d_wdcnnDF

from use_augument_data.hyper_tunnning import *
import numpy as np


def train(i=-1, times=10):
    global accuracy_list, time_list
    ## 运行在四个转速下下 （ps 这样写虽然又代码重复，但是程序不会每次都判断。就先不重构了）
    if i > 0:
        temp_list = []
        ## k 控制运行函数的数量，k=0，从第一个开始
        k = 4
        for k in range(len(function_list)):
            for n in range(len(path_list)):
                for i in range(times):
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
            for i in range(times):
                temp = function_list[k](path_list[0])
                accuracy_list[path_list[0][-3:]].append(temp[0])
                time_list[path_list[0][-3:]].append(temp[1])
            all_accuracy_list[function_list[k].__name__[6:]] = accuracy_list
            all_time_list[function_list[k].__name__[6:]] = time_list
            accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
            time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    print(all_accuracy_list)
    print(all_time_list)
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
        f.write("##################################################\n")


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
    all_accuracy_list = {"lenet5": accuracy_list, "lenet5_DF": accuracy_list, "zhao": accuracy_list,
                         "cdf_2d": accuracy_list, "wdcnn_2d": accuracy_list, "wdcnnDF_2d": accuracy_list}
    all_time_list = {"lenet5": time_list, "lenet5_DF": time_list, "zhao": time_list,
                         "cdf_2d": time_list, "wdcnn_2d": accuracy_list, "wdcnnDF_2d": accuracy_list}
    function_list = [train_lenet5, train_lenet5_DF, train_zhao, train_cdf2d, train_wdcnn, train_2d_wdcnnDF]
    train(10)



# from sklearn.manifold import TSNE
# from use_augument_data.drawing.draw_tSNE import toDF, draw_scatterplot
# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, random_state=401, metric='cosine')
# emb = tsne.fit_transform(layer_test)
# toDF(emb, y_test)
# draw_scatterplot('train_tSNE.csv')
