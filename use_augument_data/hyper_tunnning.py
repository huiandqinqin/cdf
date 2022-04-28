batch_size = 128
epochs = 20
num_classes = 10
BatchNorm = True # 是否批量归一化

data_mark = "DE"
fs = 12000
## length 暂时不该，因为cnn中input_shape 需要变，下次在优化
win_tlen = 1024 / 12000
overlap_rate = (630 / 1024) * 100
random_seed = 1

labels = {"normal": 0, "OR21": 1, "OR14": 2, "OR7": 3, "IR21": 4, "IR14": 5, "IR7": 6, "BF21": 7, "BF14": 8,
          "BF7": 9}
fault_types = labels.keys()

path_list = [r'../../data/0HP', r'../../data/1HP', r'../../data/2HP', r'../../data/3HP']

# 防止重复
# accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
# time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
# 120 overlap_rate = (24 / 1024) * 100
# 300 overlap_rate = (630 / 1024) * 100
# 900 overlap_rate = (891 / 1024) * 100c
# 1500 overlap_rate = (950 / 1024) * 100
# 12000 overlap_rate = (1017 / 1024) * 100