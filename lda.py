import codecs
import os.path
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pyLDAvis.gensim_models
import pyLDAvis
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import pickle


# 读取文本文件
def read_data(data_path="data/processed/weibo_nov_processed.txt"):
    data = []
    fp = codecs.open(data_path, 'r', encoding='utf-8')
    for line in fp:
        if line != '':
            line = line.split()
            data.append([w for w in line if len(line) > 30])
    return data


# 获取dictionary和bow矩阵
def get_bow_matrix(data):
    # 建立词典
    dictionary = corpora.Dictionary(data)
    # print(dictionary)
    # 构建bow_matrix矩阵
    corpus = [dictionary.doc2bow(text) for text in data]
    length = len(dictionary)
    bow_matrix = np.zeros((length, length))
    for i, doc in enumerate(corpus):
        for word in doc:
            bow_matrix[i, word[0]] = word[1]
    print(bow_matrix)

    return dictionary, bow_matrix


# 计算一致性得分和困惑度
def calculate_score(data, max_topic=100):
    # 建立词典
    dictionary = corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in data]
    # 更加lda计算一致性得分和困惑度
    perplexities = []
    coherence = []
    for i in range(1, max_topic, 3):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, passes=15, alpha=3, eta=0.3)
        # 计算一致性得分
        coherence_model_lda = CoherenceModel(model=model, texts=data, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence.append(coherence_lda)
        print(f"主题数：{i}，一致性得分：{coherence_lda}，", end="")

        # 计算困惑度
        perplexity = model.log_perplexity(corpus)
        perplexities.append(perplexity)
        print(f"困惑度：{perplexity}")

    # 绘制困惑度折线图
    x = range(1, max_topic, 3)  # 主题范围数量
    y1 = coherence
    y2 = perplexities
    plt.plot(x, y1, label="一致性得分", color='blue')
    plt.plot(x, y2, label="困惑度", color='red')
    plt.xlabel('主题数目')
    plt.ylabel('困惑度/一致性得分')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('主题与一致性得分/困惑度变化情况')
    plt.show()


# 训练lda模型，返回“文档-主题”矩阵和“主题-词”矩阵
def train_lda_model(data, num_topics=5, passes=200, alpha=2.5, eta=0.01):
    # 建立词典
    dictionary = corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in data]
    print(corpus[0])

    # 训练模型
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, alpha=alpha, eta=eta)

    # 可视化
    display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(display, 'lda.html')

    # 获取文档-主题矩阵
    document_topic_matrix = []
    print("\n文档-主题矩阵:")
    for doc in corpus:
        second_elements = [item[1] for item in lda_model.get_document_topics(doc)]
        print(second_elements)
        document_topic_matrix.append(np.array(second_elements))

    # 获取主题-词矩阵
    topic_word_matrix = lda_model.get_topics()
    print("\n主题-词矩阵:")
    print(topic_word_matrix)
    df_topic_word_matrix = pd.DataFrame(topic_word_matrix, columns=dictionary.values())
    print(df_topic_word_matrix)

    with open(os.path.join('data', 'interim', 'doc_topic_mat.pickle'), 'wb') as f:
        pickle.dump(document_topic_matrix, f)

    with open(os.path.join('data', 'interim', 'topic_word_mat.pickle'), 'wb') as f:
        pickle.dump(topic_word_matrix, f)

    return document_topic_matrix, topic_word_matrix


if __name__ == '__main__':
    random.seed(114514)
    data = read_data()
    print(len(data))
    selected_data = random.sample(data, 200)
    # dic, matrix = get_bow_matrix(data)
    # calculate_score(selected_data)
    train_lda_model(data, num_topics=10)
