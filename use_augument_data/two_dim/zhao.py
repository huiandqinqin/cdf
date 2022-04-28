from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, MaxPooling2D, Activation, Flatten, AveragePooling2D
from keras.models import Sequential
import time
from use_augument_data.hyper_tunnning import *
from use_augument_data.my_utils.get_train_test import get_train_test

def zhao():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',padding='same', input_shape=(32, 32, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same' ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(units=1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(units=256, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(units=10, activation='softmax'))

    # 编译模型 评价函数和损失函数相似，不过评价函数的结果不会用于训练过程中
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_zhao(path):
    start = time.time()
    x_train, x_test, y_train, y_test = get_train_test(path)
    input_shape = x_train.shape[1:]
    model = zhao()
    history = model.fit(x_train, y_train, epochs=20)
    loss, accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)
    model.summary()
    print("测试集上的损失：", loss)
    print("模型上的正确率:", accuracy)
    end = time.time()
    print("本次消耗的时间为:" + str(end - start))
    return accuracy, end - start


def draw_sne(path):
    from keras.models import load_model
    from use_augument_data.my_utils.get_layer import get_layer_output
    from sklearn.manifold import TSNE
    import pandas as pd
    import seaborn as sns
    model = load_model("zhao_df_best.h5")
    x_train, x_test, y_train, y_test = get_train_test(path)
    test = get_layer_output(model, x_test, -2)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(test)
    p_data = pd.DataFrame(columns=['x', 'y', 'label'])
    p_data.x = low_dim_embs[:, 0]
    p_data.y = low_dim_embs[:, 1]
    p_data.label = y_test
    print(p_data)
    from use_augument_data.my_utils.draw_utils import plot_with_labels as myplot
    myplot(p_data)
    loss, accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)
    print("测试集上的损失：", loss)
    print("模型上的正确率:", accuracy)




if __name__ == '__main__':
    accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    # temp_list = []
    # for n in range(len(path_list)):
    #     for i in range(10):
    #         temp = lennet5(path_list[n])
    #         accuracy_list[path_list[n][-3:]].append(temp[0])
    #         time_list[path_list[n][-3:]].append(temp[1])
    draw_sne(path_list[3])
