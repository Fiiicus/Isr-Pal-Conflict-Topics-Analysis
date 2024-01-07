import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import imageio
import lda


def cluster():
    num_topic, num_cluster, num_keywords = 15, 5, 30
    data = lda.read_data()
    text_topic_matrix, topic_word_matrix, words = lda.train_lda_model(data, num_topics=num_topic)
    chi = np.empty((num_cluster, num_topic))
    for i in range(len(text_topic_matrix)):
        if len(text_topic_matrix[i]) < 15:
            dimensions_to_pad = num_topic - len(text_topic_matrix[i])
            padding = (0, dimensions_to_pad)
            text_topic_matrix[i] = np.pad(text_topic_matrix[i], padding, mode='constant', constant_values=0)

    # 为了画图
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(text_topic_matrix)
    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(X_pca)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=3, color='r')
    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # 为了实验效果
    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(text_topic_matrix)
    labels = kmeans.labels_

    clusters = {}
    doc_cluster = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        vect = np.where(num_topic * text_topic_matrix[i] < 1, 0, 1)
        clusters[label].append(vect)
        doc_cluster[i] = label

    m = []
    for i in range(num_cluster):
        m.extend(np.vstack(clusters[i]).tolist())
    m = np.array(m)

    n = len(m)  # 文档数
    for i in range(num_cluster):
        mi = np.vstack(clusters[i])  # 第i类的所有文档主题关系矩阵
        for j in range(num_topic):
            a = np.sum(mi[:, j])  # 第i类中第j个主题的数目
            b = len(mi) - a  # 第i类中没有第j个主题的数目
            c = np.sum(m[:, j]) - a  # 除第i类外包含第j个主题的文档数
            d = n - len(mi) - c  # 除第i类外不包含第j个主题的文档数
            h = n * (a * d - b * c) ** 2
            k = (a + c) * (b + d) * (a + b) * (c + d)
            chi[i][j] = h / k

    res = chi @ topic_word_matrix
    for i in range(num_cluster):
        res[i] /= np.sum(res[i])

    for i in range(num_cluster):
        sorted_chi = np.argsort(res[i])[::-1]
        word_prob = {}
        for j in range(num_keywords):
            word_prob[words[sorted_chi[j]]] = res[i][sorted_chi[j]]
        print(word_prob)
        mask = imageio.imread(r'bbll.jpeg')
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              font_path='NotoSansSC-Black.ttf').generate_from_frequencies(word_prob)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    return doc_cluster


if __name__ == '__main__':
    cluster()
