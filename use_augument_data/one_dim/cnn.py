import time
from keras.layers import Dense, Conv1D, BatchNormalization, AveragePooling1D, Activation, Flatten
from keras.models import Sequential
from keras.regularizers import l2
from use_augument_data.hyper_tunnning import *
from use_augument_data.my_utils.get_train_test import get_train_test_1dim

from use_augument_data.drawing.draw_confusion_matrix import confu_matrix
fault_types = labels.keys()


def cnn():
    model = Sequential()

    # 第一层卷积
    model.add(
        Conv1D(filters=32, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(1e-4),
               input_shape=(1024, 1)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(AveragePooling1D())

    model.add(
        Conv1D(filters=16, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(1e-4),
               input_shape=(1024, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D())

    # 从卷积到全连接需要展平
    model.add(Flatten())

    # 添加全连接层，共100个单元，激活函数为ReLU
    model.add(Dense(units=120, activation='relu', kernel_regularizer=l2(1e-4)))

    model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))

    # 编译模型
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_cnn(path):
    start = time.time()
    x_train, x_test, y_train, y_test = get_train_test_1dim(path)
    input_shape = x_train.shape[1:]
    model = cnn()
    # 开始模型训练 validation_data=(x_valid, y_valid)
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, shuffle=True)

    # 评估模型
    model.summary()
    # train and evaluate
    y_pred = model.predict(x_test)
    confu_matrix(y_test, y_pred)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    end = time.time()
    print("Test accuracy:", test_acc)
    return test_acc, end - start

if __name__ == '__main__':
    accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    # 获得data 下的数据
    # temp_list = []
    # for n in range(len(path_list)):
    #     for i in range(10):
    #         temp = train_cnn(path_list[n])
    #         accuracy_list[path_list[n][-3:]].append(temp[0])
    #         time_list[path_list[n][-3:]].append(temp[1])
    # print(accuracy_list)
    # print(time_list)
    train_cnn(path_list[0])

