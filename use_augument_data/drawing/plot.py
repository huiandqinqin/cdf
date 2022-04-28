# Import seaborn
import numpy as np
import seaborn as sns
import pandas as pd
import pylab as plt

# Apply the default theme
sns.set_theme()


def draw_stript():
    acc = pd.read_csv('../results_log/result.csv')
    acc = acc[(acc['dim'] != 1) | (acc['sort'] == 'DF')]
    # da = acc.iloc[:10]
    da = acc.iloc[:144]
    ## 散点图
    # sns.stripplot(data=acc, x="sort", y="accuracy", jitter=0.2, hue="power")
    ## 盒图
    sns.catplot(data=da, x='length', y='accuracy', hue='sort', kind='bar', col='power', ci=None)



def draw_dim():
    acc = pd.read_csv('../results_log/result.csv')
    acc = acc[ (acc['power'] == '3hp') & (acc['length'] == 300) & (acc['sort'] != 'DF')].sort_values(by=['sort'])
    dim1 = acc[ acc['dim'] == 1].sort_values(by=['sort'])
    dim2 = acc[ acc['dim'] == 2].sort_values(by=['sort'])
    # acc = acc[ (acc['length'] == 300)]
    # acc = acc[ (acc['sort'] != 'DF')]
    print(dim1)
    print(dim2)
    print(np.array(dim2['accuracy']) - np.array(dim1['accuracy']) )
    fig, ax = plt.subplots()
    g = sns.barplot(data=acc, ax= ax, x='sort', y='accuracy', hue='sort')
    g.set_yticks([60,70,80,90,100])
    g.set_xticks([])
    g.despine(left=True)
    plt.legend(loc=4, prop={'size': 8})
    change_width(ax, .35)
    # g.set_legend(loc=4, prop={'size': 6})
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


draw_dim()