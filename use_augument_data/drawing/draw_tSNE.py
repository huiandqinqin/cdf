from openTSNE import TSNE
from use_augument_data.hyper_tunnning import path_list
import numpy as np
from sklearn.manifold import TSNE

from use_augument_data.my_utils.get_train_test import get_train_test
x_train, x_test, y_train, y_test = get_train_test(path_list[0])
x_train = x_train.reshape(x_train.shape[0], 1024)
x_test = x_test.reshape(x_test.shape[0], 1024)


print(x_train.shape)

# tsne = manifold.TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# embedding = np.array(TSNE(perplexity=80, n_iter=5000, metric='cosine', initialization='pca', random_state=4).fit(x_test))
tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=5000, metric='cosine', random_state=401, early_exaggeration=6)
embedding = tsne.fit_transform(x_test)
def toDF(data, label):
    import pandas as pd
    x = embedding[:,0]
    y = embedding[:,1]
    data = {'x':x , 'y': y, 'sort': label}
    df = pd.DataFrame(data)
    df.to_csv("train_tSNE.csv")
    print("save successfully")


def draw_scatterplot(path):
    import pandas as pd
    import seaborn as sns
    r_data = pd.read_csv(path)
    sns.scatterplot(data=r_data, x='x', y='y', hue='sort', palette=sns.color_palette("tab10") )

def generate_random_data():
    pass
toDF(embedding, y_test)
draw_scatterplot('train_tSNE.csv')

# X_2d = tsne.fit(x_train)
