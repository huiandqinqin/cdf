from keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, Activation, Flatten, AveragePooling2D
from keras.models import Sequential
from keras.regularizers import l2

import time
from use_augument_data.hyper_tunnning import *
from use_augument_data.my_utils.get_train_test import get_train_test



def lenet5():
    model = Sequential()
    # 第一层卷积
    model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='same',input_shape=(32, 32, 1)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(AveragePooling2D())

    model.add(
        Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D())

    # 从卷积到全连接需要展平
    model.add(Flatten())

    # 添加全连接层，共100个单元，激活函数为ReLU
    model.add(Dense(units=120, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dense(units=10, activation='softmax'))

    # 编译模型
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def lenet20():
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(20, 20), activation='relu', input_shape=(32, 32, 1)))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))

    # model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    # 编译模型 评价函数和损失函数相似，不过评价函数的结果不会用于训练过程中
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
def train_lenet5(path):
    start = time.time()
    x_train, x_test, y_train, y_test = get_train_test(path)
    model = lenet5()
    history = model.fit(x_train, y_train, epochs=20)
    loss, accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)
    print("测试集上的损失：", loss)
    print("模型上的正确率:", accuracy)
    end = time.time()
    print("本次消耗的时间为:" + str(end - start))
    return accuracy, end - start

if __name__ == '__main__':
    accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    train_lenet5(path_list[0])
