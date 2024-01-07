import random

import numpy as np
from matplotlib import pyplot as plt
from pyLDAvis import lda_model
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pyLDAvis
import pyLDAvis.lda_model

# 要处理的月份
month = 'otc'
# 要处理文本的路径
input_filepath = f'data/processed/weibo_{month}_processed.txt'


# 读取文本
def read_docs():
    with open(input_filepath, 'r', encoding='utf8') as f:
        docs = [line.strip() for line in f]
    return docs


# 将文本向量化（词频矩阵）
def countVectorize(docs):
    # 使用CountVectorizer进行词频统计
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    print(X.shape)
    return X


def TFIDFVectorize(docs):
    # 使用CountVectorizer进行词频统计
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    # 类调用
    transformer = TfidfTransformer()

    # 将词频矩阵X统计成TF-IDF值
    tfidf = transformer.fit_transform(X)
    return tfidf


# 绘制困惑度随主题数变化的曲线
def plot_perplexity(X):
    # 定义主题数的范围
    topics_range = range(2, 20)

    # 存储每个主题数对应的模型的困惑度
    perplexity_list = []

    # 对每个主题数，训练一个LDA模型，然后计算其困惑度
    for n_topics in topics_range:
        lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online')
        lda.fit(X)
        perplexity = lda.perplexity(X)
        print(f'主题数：{n_topics}\t困惑度：{perplexity}')
        perplexity_list.append(perplexity)

    # 绘制困惑度随主题数变化的曲线
    plt.figure(figsize=(10, 5))
    plt.plot(topics_range, perplexity_list)
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs Number of Topics')
    plt.show()


# 绘制困惑度随alpha变化的曲线
def plot_alpha(X, n_topics):
    # 定义alpha的范围
    alpha_range = np.linspace(0, 1, 50)

    # 存储每个alpha对应的模型的困惑度
    perplexity_list = []

    # 对每个alpha，训练一个LDA模型，然后计算其困惑度
    for alpha in alpha_range:
        lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online', doc_topic_prior=alpha)
        lda.fit(X)
        perplexity = lda.perplexity(X)
        print(f'alpha：{alpha}\t困惑度：{perplexity}')
        perplexity_list.append(perplexity)

    # 绘制困惑度随alpha变化的曲线
    plt.figure(figsize=(10, 5))
    plt.plot(alpha_range, perplexity_list)
    plt.xlabel('value of alpha')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs value of alpha')
    plt.show()


# 绘制困惑度随eta变化的曲线
def plot_eta(X, n_topics):
    # TODO
    pass


# 输入分词后的文本，返回lda后的两个矩阵
def do_lda(docs, n_topics, alpha, eta=0.1):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)

    # 创建LDA模型实例
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online', doc_topic_prior=alpha,
                                    topic_word_prior=eta)

    # 使用LDA模型拟合数据
    lda.fit(X)

    # 假设lda_model是你已经训练好的LDA模型，doc_term_matrix是文档-词项矩阵，vectorizer是用于生成doc_term_matrix的向量化器
    data = pyLDAvis.lda_model.prepare(lda, X, vectorizer)
    pyLDAvis.save_html(data, f'{month}_lda_sklearn.html')

    doc_topic_matrix = lda.transform(X)
    topic_word_matrix = lda.components_
    print(f'shape of doc_topic_matrix: {doc_topic_matrix.shape}')
    print(f'shape of topic_word_matrix: {topic_word_matrix.shape}')
    return doc_topic_matrix, topic_word_matrix, vectorizer.get_feature_names_out()


if __name__ == '__main__':
    # 随机种子
    # random.seed(114514)
    docs = read_docs()
    # # sampled_docs = random.sample(docs, len(docs)//10)
    # plot_perplexity(countVectorize(docs))
    # plot_alpha(countVectorize(docs), 10)
    do_lda(docs, 7, 0.15)
