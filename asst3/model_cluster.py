from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    with open('../name.txt', 'r', encoding='utf8') as f:
        names = f.readline().split(' ')
    model = Word2Vec.load('model1.model')
    names = [name for name in names if name in model.wv]
    #获取名字对应的词向量
    name_vectors = [model.wv[name] for name in names]
    #创建一个TSNE对象，并使用fit_transform方法将词向量降维到二维空间。
    tsne = TSNE()
    embedding = tsne.fit_transform(np.array(name_vectors))
    n = 5
    label = KMeans(n).fit(embedding).labels_
    plt.title('kmeans聚类结果')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    colors = ['green', 'yellow', 'cyan', 'blue', 'magenta']
    for i in range(len(label)):
        plt.plot(embedding[i][0], embedding[i][1], 'o', color=colors[label[i]])
        plt.annotate(names[i], xy=(embedding[i][0], embedding[i][1]), xytext=(embedding[i][0]+0.1, embedding[i][1]+0.1), fontsize=6)
    plt.show()