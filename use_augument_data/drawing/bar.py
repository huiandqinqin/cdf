import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()
sns.set_style('dark')
sns.set_palette(sns.color_palette("Set2"))

# sns.palplot(sns.color_palette('Blues'))
# ,lenet5_120,lenet5_DF_120,zhao_120,zhao_DF_120,\
#  lenet5_300,lenet5_DF_300,zhao_300,zhao_DF_300,\
#  lenet5_900,lenet5_DF_900,zhao_900,zhao_DF_900,\
#  lenet5_1500,lenet5_DF_1500,zhao_1500,zhao_DF_1500,\
#  cnn_120,cnnDF_120,deep_forest_120,wdcnn_120,wdcnnDF_120,\
#  cnn_300,cnnDF_300,deep_forest_300,wdcnn_300,wdcnnDF_300,\
#  cnn_900,cnnDF_900,deep_forest_900,wdcnn_900,wdcnnDF_900,\
#  cnn_1500,cnnDF_1500,deep_forest_1500,wdcnn_1500,wdcnnDF_1500

def read_csv_f1(path, start_hp=0, end_hp=1):
    raw_data = pd.read_csv(path)
    # for i in range(10):
    #     print(raw_data.iloc[0:1, (4 * i + 1 ): (4 * i + 5)])
    ## 1 先按子样本长度分，然后按一二维分
    z1 = raw_data[['cnn_1_120','wdcnn_1_120','deepforest_120','cnnDF_1_120','wdcnnDF_1_120','cnn_2_120','cnnDF_2_120', 'proposed_2_300', 'proposedDF_2_300']].iloc[start_hp:end_hp]
    z2 = raw_data[['cnn_1_300','wdcnn_1_300','deepforest_300','cnnDF_1_300','wdcnnDF_1_300','cnn_2_300','cnnDF_2_300', 'proposed_2_300', 'proposedDF_2_300']].iloc[start_hp:end_hp]
    z3 = raw_data[['cnn_1_900','wdcnn_1_900','deepforest_900','cnnDF_1_900','wdcnnDF_1_900','cnn_2_900','cnnDF_2_900', 'proposed_2_900', 'proposedDF_2_900']].iloc[start_hp:end_hp]
    z4 = raw_data[['cnn_1_1500','wdcnn_1_1500','deepforest_1500','cnnDF_1_1500','wdcnnDF_1_1500','cnn_2_1500','cnnDF_2_1500', 'proposed_2_1500', 'proposedDF_2_1500']].iloc[start_hp:end_hp]

    # print(list(z1))
    np1 = np.array(z1)
    np2 = np.array(z2)
    return z1, z2, z3, z4
def read_csv_f2(path, raw_matrix=False, start_hp=0, end_hp=1):
    ## 2
    raw_data = pd.read_csv(path)
    row1 = raw_data[['cnn_1_120', 'cnn_1_300', 'cnn_1_900', 'cnn_1_1500']]
    row2 = raw_data[['wdcnn_1_120', 'wdcnn_1_300', 'wdcnn_1_900', 'wdcnn_1_1500']]
    row3 = raw_data[['deepforest_120', 'deepforest_300', 'deepforest_900', 'deepforest_1500']]
    row4 = raw_data[['cnnDF_1_120', 'cnnDF_1_300', 'cnnDF_1_900', 'cnnDF_1_1500']]
    row5 = raw_data[['wdcnnDF_1_120', 'wdcnnDF_1_300', 'wdcnnDF_1_900', 'wdcnnDF_1_1500']]
    row6 = raw_data[['cnn_2_120', 'cnn_2_300', 'cnn_2_900', 'cnn_2_1500']]
    row7 = raw_data[['cnnDF_2_120', 'cnnDF_2_300', 'cnnDF_2_900', 'cnnDF_2_1500']]
    row8 = raw_data[['proposed_2_120', 'proposed_2_300', 'proposed_2_900', 'proposed_2_1500']]
    row9 = raw_data[['proposedDF_2_120', 'proposedDF_2_300', 'proposedDF_2_900', 'proposedDF_2_1500']]
    if not raw_matrix:
        z1 = row1.iloc[start_hp:end_hp]
        z2 = row2.iloc[start_hp:end_hp]
        z3 = row3.iloc[start_hp:end_hp]
        z4 = row4.iloc[start_hp:end_hp]
        z5 = row5.iloc[start_hp:end_hp]
        z6 = row6.iloc[start_hp:end_hp]
        z7 = row7.iloc[start_hp:end_hp]
        z8 = row8.iloc[start_hp:end_hp]
        z9 = row9.iloc[start_hp:end_hp]
        return z1, z2, z3, z4, z5, z6, z7, z8, z9
    else:
        return row1, row2, row3, row4, row5, row6, row7, row8, row9


def draw_bar_f1(data_):
    x_data, y_data, z_data, c_data = data_
    width = 1  # the width of the bars
    labels = ['1d_cnn', '1d_wdcnn', 'df', '1dcnnDF', '1dwdcnnDF', '2d_cnn', '2d_cnnDF', '2d_proposed', '2d_proposedDF']
    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots()
    rects1 = ax.bar(4 * x - width, height=np.around(np.array(x_data)[0]), label='120', width=0.5)
    rects2 = ax.bar(4 * x - width/2, height=np.around(np.array(y_data)[0]), label='300', width=0.5)
    rects3 = ax.bar(4 * x, height=np.around(np.array(z_data)[0]), label='900', width=0.5)
    rects4 = ax.bar(4 * x + width/2, height=np.around(np.array(c_data)[0]), label='1500', width=0.5)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Different sample')
    ax.set_xticks(4 * x)
    ax.set_xticklabels(labels)
    # ax.set_yticks([90, 100, 1])
    ax.legend(loc=4)

    # ax.bar_label(rects1, padding=5, label_type='center')
    # ax.bar_label(rects2, padding=5, label_type='center')
    # ax.bar_label(rects3, padding=5, label_type='center')
    # ax.bar_label(rects4, padding=5, label_type='center')
    fig.tight_layout()
    plt.show()


def draw_bar_f2(data_):
    z1, z2, z3, z4, z5, z6, z7, z8, z9 = data_
    width = 9  # the width of the bars
    labels = ['120', '300', '900', '1500']
    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots()
    rects1 = ax.bar(12 * x - 5, height=np.around(np.array(z1)[0]), label='cnn', width=0.8)
    rects2 = ax.bar(12 * x - 4, height=np.around(np.array(z2)[0]), label='wdcnn', width=0.8)
    rects3 = ax.bar(12 * x - 3, height=np.around(np.array(z3)[0]), label='df', width=0.8)
    rects4 = ax.bar(12 * x - 2, height=np.around(np.array(z4)[0]), label='cnnDF', width=0.8)
    rects5 = ax.bar(12 * x - 1, height=np.around(np.array(z1)[0]), label='wdcnnDF', width=0.8)
    rects6 = ax.bar(12 * x , height=np.around(np.array(z2)[0]), label='lenet5', width=0.8)
    rects7 = ax.bar(12 * x + 1, height=np.around(np.array(z3)[0]), label='lenet5DF', width=0.8)
    rects8 = ax.bar(12 * x + 2, height=np.around(np.array(z4)[0]), label='zhao', width=0.8)
    rects9 = ax.bar(12 * x + 3, height=np.around(np.array(z4)[0]), label='zhaoDF', width=0.8)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Different sample')
    ax.set_xticks(12 * x)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0,101,10))
    ax.set_yticklabels(['0', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    ax.legend(loc=4)

    # ax.bar_label(rects1, padding=5, label_type='center')
    # ax.bar_label(rects2, padding=5, label_type='center')
    # ax.bar_label(rects3, padding=5, label_type='center')
    # ax.bar_label(rects4, padding=5, label_type='center')

    fig.tight_layout()
    plt.show()
# 画四个工况下（0，1，2，3hp）每个模型下的柱状图
def draw_bar_f2_allhp(data_):
    row1, row2, row3, row4, row5, row6, row7, row8, row9 = data_
    width = 9  # the width of the bars
    labels = ['120', '300', '900', '1500']
    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    for i in range(4):
        ax[i].bar(12 * x - 5, height=np.around(np.array(row1.iloc[i:i+1])[0]), label='1d_cnn', width=0.8)
        ax[i].bar(12 * x - 4, height=np.around(np.array(row2.iloc[i:i+1])[0]), label='1d_wdcnn', width=0.8)
        ax[i].bar(12 * x - 3, height=np.around(np.array(row3.iloc[i:i+1])[0]), label='DF', width=0.8)
        ax[i].bar(12 * x - 2, height=np.around(np.array(row4.iloc[i:i+1])[0]), label='1d_cnnDF', width=0.8)
        ax[i].bar(12 * x - 1, height=np.around(np.array(row5.iloc[i:i+1])[0]), label='1d_wdcnnDF', width=0.8)
        ax[i].bar(12 * x, height=np.around(np.array(row6.iloc[i:i+1])[0]), label='2d_cnn', width=0.8)
        ax[i].bar(12 * x + 1, height=np.around(np.array(row7.iloc[i:i+1])[0]), label='2d_cnnDF', width=0.8)
        ax[i].bar(12 * x + 2, height=np.around(np.array(row8.iloc[i:i+1])[0]), label='2d_proposed', width=0.8)
        ax[i].bar(12 * x + 3, height=np.around(np.array(row9.iloc[i:i+1])[0]), label='2d_proposedDF', width=0.8)

        ax[i].set_ylabel('Accuracy')
        ax[i].set_title('{}hp'.format(i))
        ax[i].set_xticks(12 * x)
        ax[i].set_xticklabels(labels)
        ax[i].set_yticks(np.arange(0, 101, 10))
        ax[i].set_yticklabels(['0', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
        ax[i].legend(loc=4, prop={'size': 6})
    # Add some text for labels, title and custom x-axis tick labels, etc.

    # ax.bar_label(rects1, padding=5, label_type='center')
    # ax.bar_label(rects2, padding=5, label_type='center')
    # ax.bar_label(rects3, padding=5, label_type='center')
    # ax.bar_label(rects4, padding=5, label_type='center')

    fig.tight_layout()
    plt.show()
# 画四个工况下（0，1，2，3hp）每个模型 2维的柱状图
def draw_bar_f2_allhp_2dim(data_):
    row1, row2, row3, row4, row5, row6, row7, row8, row9 = data_
    width = 9  # the width of the bars
    labels = ['120', '300', '900', '1500']
    x = np.arange(len(labels))  # the label locations
    fig, ax = plt.subplots(2, 2, figsize=(13, 6.2))
    for i in range(2):
        for j in range(2):
            iloc = int('{}{}'.format(i, j), 2)
            rects1 = ax[i][j].bar(10 * x - 3.2, height=np.around(np.array(row3.iloc[iloc:iloc+1])[0], 2), label='DF', width=1.6)
            rects2 =ax[i][j].bar(10 * x - 1.6, height=np.around(np.array(row6.iloc[iloc:iloc+1])[0], 2), label='Lenet5', width=1.6)
            rects3 = ax[i][j].bar(10 * x , height=np.around(np.array(row7.iloc[iloc:iloc+1])[0], 2), label='Lenet5DF', width=1.6)
            rects4 = ax[i][j].bar(10 * x + 1.6, height=np.around(np.array(row8.iloc[iloc:iloc+1])[0], 2), label='cnn', width=1.6)
            rects5 = ax[i][j].bar(10 * x + 3.2, height=np.around(np.array(row9.iloc[iloc:iloc+1])[0], 2), label='CDF', width=1.6)

            ax[i][j].set_ylabel('Acc')
            ax[i][j].set_title('{}hp'.format(iloc))
            ax[i][j].set_xticks(10 * x)
            ax[i][j].set_xticklabels(labels)
            ax[i][j].set_yticks(np.arange(0, 101, 10))
            ax[i][j].set_yticklabels(['0', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
            ax[i][j].legend(loc=4, prop={'size': 6})

            ax[i][j].bar_label(rects1, padding=-1, label_type='edge',  fontsize=8)
            ax[i][j].bar_label(rects2, padding=-1, label_type='edge',  fontsize=8)
            ax[i][j].bar_label(rects3, padding=-1, label_type='edge',  fontsize=8)
            ax[i][j].bar_label(rects4, padding=-1, label_type='edge',  fontsize=8)
            ax[i][j].bar_label(rects5, padding=-1, label_type='edge', fontsize=8)

    fig.tight_layout()
    plt.show()



data = read_csv_f2('../results_log/all_results.csv', True)
draw_bar_f2_allhp_2dim(data)
