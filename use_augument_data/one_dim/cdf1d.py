from deepforest import CascadeForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import time
from use_augument_data.my_utils.get_layer import get_layer_output
from use_augument_data.my_utils.get_train_test import get_train_test_1dim
from use_augument_data.hyper_tunnning import *
from use_augument_data.one_dim.zhao1d import zhao_1d



def train_cdf1dDF(path):
    start = time.time()
    x_train, x_test, y_train, y_test = get_train_test_1dim(path)

    model = zhao_1d()
    history = model.fit(x_train, y_train, epochs=20)
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
    # 获得data 下的数据
    i = 0
    if i > 10:
        temp_list = []
        for n in range(len(path_list)):
            for i in range(10):
                temp = train_cdf1dDF(path_list[n])
                accuracy_list[path_list[n][-3:]].append(temp[0])
                time_list[path_list[n][-3:]].append(temp[1])
        print(accuracy_list)
        print(time_list)
    else:
        for i in range(1):
            temp = train_cdf1dDF(path_list[0])
            accuracy_list[path_list[0][-3:]].append(temp[0])
            time_list[path_list[0][-3:]].append(temp[1])
