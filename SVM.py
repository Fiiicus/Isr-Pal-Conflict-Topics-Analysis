import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

import LDA_sklearn
import kmeans


def get_train(indices, BOW):
    data = BOW[[_[0] for _ in indices]]
    label = [_[1] for _ in indices]
    return data, label


def get_test(filepath, vocab):
    with open(filepath, 'r', encoding='utf8') as f:
        text_list = [_ for _ in f]
        # 创建一个CountVectorizer对象
        vectorizer = CountVectorizer(vocabulary=vocab)
        # 使用CountVectorizer拟合并转换文本数据，得到BOW矩阵
        bow_matrix = vectorizer.fit_transform(text_list)
        # 将BOW矩阵转换为稠密数组
        bow_matrix = bow_matrix.toarray()
        print(bow_matrix)
    return bow_matrix


docs = LDA_sklearn.read_docs()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
y = list(kmeans.cluster().values())

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train[0], y_train.shape)

# 创建一个SVC分类器，并启用概率估计
svm_classifier = SVC(kernel='linear', probability=True)

# 在训练数据上训练模型
svm_classifier.fit(X_train, y_train)

# 使用训练后的模型进行预测
y_pred = svm_classifier.predict(X_test)

# 计算分类准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'分类准确率: {accuracy}')

# 获取分类的置信度（概率）
confidence_scores = svm_classifier.predict_proba(X_test)
print('分类的置信度:')
print(confidence_scores)

# 输出测试集中每个样本的分类结果
print('测试集中每个样本的分类结果:')
for i in range(X_test.shape[0]):
    print(f'样本 {i + 1}: 预测类别为 {y_pred[i]}, 置信度为 {max(confidence_scores[i]):.2f}')

# 输出混淆矩阵
confusion = confusion_matrix(y_test, y_pred)
print('混淆矩阵:')
print(confusion)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('predict')
plt.ylabel('true')
plt.title('confusion_matrix')
plt.show()

words = vectorizer.get_feature_names_out()
X = get_test("data/processed/weibo_new_processed.txt", words)
predict = svm_classifier.predict(X)
confidence_scores = svm_classifier.predict_proba(X)
print('新数据中每个样本的分类结果:')
count = 0
for i in range(X.shape[0]):
    if max(confidence_scores[i]) < 0.5:
        count = count + 1
    print(f'样本 {i + 1}: 预测类别为 {predict[i]}, 置信度为 {max(confidence_scores[i]):.2f}')
print(f'{count / X.shape[0]:.2f}')
