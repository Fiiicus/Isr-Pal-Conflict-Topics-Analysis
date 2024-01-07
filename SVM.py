from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

import LDA_sklearn
import kmeans

# 新话题阈值
threshold = 0.6


def get_train(X_train, y_train):
    # 创建一个SVC分类器，并启用概率估计
    svm_classifier = SVC(kernel='linear', probability=True)
    # 在训练数据上训练模型
    svm_classifier.fit(X_train, y_train)
    return svm_classifier


def get_test(svm_classifier, X_test, y_test):
    y_pred = svm_classifier.predict(X_test)
    # 计算分类准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n分类准确率: {accuracy}')

    # 获取分类的置信度（概率）
    confidence_scores = svm_classifier.predict_proba(X_test)
    print('\n分类的置信度:')
    print(confidence_scores)

    # 输出测试集中每个样本的分类结果
    print('\n测试集中每个样本的分类结果:')
    for i in range(X_test.shape[0]):
        print(f'样本 {i + 1}: 预测类别为 {y_pred[i]}, 置信度为 {max(confidence_scores[i]):.2f}')

    # 输出混淆矩阵
    confusion = confusion_matrix(y_test, y_pred)
    print('\n混淆矩阵:')
    print(confusion)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('predict')
    plt.ylabel('true')
    plt.title('confusion_matrix')
    plt.show()


def predict_new(svm_classifier, file_path, words):
    with open(file_path, 'r', encoding='utf8') as f:
        text_list = [_ for _ in f]
        # 创建一个CountVectorizer对象
        vectorizer = CountVectorizer(vocabulary=words)
        # 使用CountVectorizer拟合并转换文本数据，得到BOW矩阵
        bow_matrix = vectorizer.fit_transform(text_list)
        # 将BOW矩阵转换为稠密数组
        X = bow_matrix.toarray()
        # print(bow_matrix)
    predict = svm_classifier.predict(X)
    confidence_scores = svm_classifier.predict_proba(X)
    print('新数据中每个样本的分类结果:')
    count = 0
    for i in range(X.shape[0]):
        if max(confidence_scores[i]) < threshold:
            count = count + 1
        print(f'样本 {i + 1}: 预测类别为 {predict[i]}, 置信度为 {max(confidence_scores[i]):.2f}')
    print(f'{count / X.shape[0]:.2f}')


# 获取有监督数据集X和y
docs = LDA_sklearn.read_docs()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
y = list(kmeans.cluster().values())

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练SVM分类器
svm_classifier = get_train(X_train, y_train)

# 测试集预测
get_test(svm_classifier, X_test, y_test)

# 测试新数据
new_data_path = "data/processed/weibo_new_processed.txt"
predict_new(svm_classifier, new_data_path, vectorizer.get_feature_names_out())
