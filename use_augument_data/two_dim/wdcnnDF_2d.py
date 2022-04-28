# coding=UTF-8
import time

from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier
from use_augument_data.hyper_tunnning import *
from use_augument_data.my_utils.get_layer import get_layer_output
from use_augument_data.my_utils.get_train_test import get_train_test
from use_augument_data.two_dim.wdcnn_2d import wdcnn






def train_2d_wdcnnDF(path):
    start = time.time()
    x_train, x_test, y_train, y_test = get_train_test(path)

    # model = load_model("zhao_df.h5")
    # model = my_model()
    # history = model.fit(x_train, y_train, epochs=20)
    # model.save("zhao_df.h5")
    # model = load_model("zhao_df.h5")
    model = wdcnn()
    model.fit(x_train, y_train, epochs=20)
    layer_train = get_layer_output(model, x_train, index=-2)
    layer_test = get_layer_output(model, x_test, index=-2)

    print("layer_train's shape{0}".format(layer_train.shape))
    print("layer_test's shape{0}".format(layer_test.shape))

    model = CascadeForestClassifier(n_jobs=-1, n_estimators=4, n_trees=300, max_layers=10)
    model.fit(layer_train, y_train)

    # train and evaluate
    y_pred = model.predict(layer_test)
    acc = accuracy_score(y_test, y_pred) * 100  # classification accuracy
    print("accuracy", acc)
    end = time.time()
    print("本次消耗的时间为:" + str(end - start))
    return acc, end - start


if __name__ == '__main__':
    accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    i = 0
    ## 运行在四个转速下下
    if i > 10:
        temp_list = []
        for n in range(len(path_list)):
            for i in range(10):
                temp = train_2d_wdcnnDF(path_list[n])
                accuracy_list[path_list[n][-3:]].append(temp[0])
                time_list[path_list[n][-3:]].append(temp[1])
    ## 运行在某一个转速下
    else:
        for i in range(1):
            temp = train_2d_wdcnnDF(path_list[0])
            accuracy_list[path_list[0][-3:]].append(temp[0])
            time_list[path_list[0][-3:]].append(temp[1])
    print(accuracy_list)
    print(time_list)