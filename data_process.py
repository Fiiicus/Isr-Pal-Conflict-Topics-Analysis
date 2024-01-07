import re
import csv
import random
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def lower(datalist):
    for i in range(len(datalist)):
        datalist[i] = datalist[i].lower()


def process(datalist):
    result_list = []
    for i in range(len(datalist)):
        words = word_tokenize(datalist[i])  # 分词
        words_without_non_alpha = [word for word in words if word.isalpha()]     # 过滤非英文单词
        lower(words_without_non_alpha)
        filtered = [word for word in words_without_non_alpha if word not in stop_words]   # 去除停用词
        if len(filtered) > 30:  # 只取长度>30的文档
            result_list.append(filtered)
    return result_list


def join(datalist):
    res = []
    for items_tmp in datalist:
        tmp = ""
        for item_tmp in items_tmp:
            tmp += item_tmp
            tmp += " "
        res.append(tmp)
    return res


def singularize_words(word_list):
    singularized_words = []
    lemmatizer = WordNetLemmatizer()
    for words in word_list:
        singularized_words.append([lemmatizer.lemmatize(word, pos='n') for word in words])
    print(f"singularized:{singularized_words}")
    return singularized_words


def tf_idf_filter(datalist, tfidf_threshold, length_threshold=100000):
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer()

    # 复数单数化处理
    singular = singularize_words(datalist)

    # 将单词拼接成文档，对文档数据进行拟合和转换
    tfidf_matrix = vectorizer.fit_transform(join(singular))
    # 获取特征名（即关键词）
    feature_names = vectorizer.get_feature_names_out()
    # 选出TF-IDF值小于阈值的词
    filter_words = [feature_names[j] for j in range(tfidf_matrix.shape[1]) if
                    np.max(tfidf_matrix[:, j]) <= tfidf_threshold]

    # 过滤TF-IDF值小于阈值的词
    # filtered_words_tmp = []
    tfidf_filtered = [word for word in singular if word not in filter_words]
    length_filtered = [sublist for sublist in tfidf_filtered if len(sublist) >= length_threshold]
    return length_filtered


def read_stopwords(filepath):
    file = open(filepath, 'r', encoding='UTF-8')
    res = []
    for line_tmp in file:
        line_tmp = line_tmp[0:len(line_tmp) - 1]
        res.append(line_tmp)
    file.close()
    return res


def read_txt(filepath):
    res = []
    file = open(filepath, 'r', encoding='UTF-8')
    lines_tmp = file.readlines()
    for line_tmp in lines_tmp:
        res.append(line_tmp.replace("\n", ""))
    return res


def remove_urls_and_emojis_from_list(documents):
    cleaned_documents = []

    for document in documents:
        # 去除网址
        text_without_urls = re.sub(r'http\S+|www\S+|https\S+', '', document, flags=re.MULTILINE)

        # 去除表情符号
        text_without_emojis = re.sub(r'[\U00010000-\U0010ffff]', '', text_without_urls)

        cleaned_documents.append(text_without_emojis)

    return cleaned_documents


def write(filepath, datalist):
    file = open(filepath, 'w', encoding='UTF-8')
    file.close()
    file = open(filepath, 'a', encoding='UTF-8')
    for items in datalist:
        for item in items:
            file.write(item + " ")
        file.write("\n")
    file.close()


threshold = 10

# 读取评论数据并去除url和emoji
comments_sep = remove_urls_and_emojis_from_list(read_txt(".\\input\\sep_comments.txt"))
comments_oct = remove_urls_and_emojis_from_list(read_txt(".\\input\\oct_comments.txt"))
comments_nov = remove_urls_and_emojis_from_list(read_txt(".\\input\\nov_comments.txt"))
comments_dec = remove_urls_and_emojis_from_list(read_txt(".\\input\\dec_comments.txt"))
print("eliminate urls and emojis done")

# 读停用词
stop_words = read_stopwords("stopwords.txt")

# 分词，过滤停用词
list_sep = process(comments_sep)
print(f"Sep: filter stopwords done. length: {len(list_sep)}")
#
# selected_comments_oct = random.sample(comments_oct, 1000)
list_oct = process(comments_oct)
print(f"Oct: filter stopwords done. length: {len(list_oct)}")
#
# selected_comments_nov = random.sample(comments_nov, 1000)
list_nov = process(comments_nov)
print(f"Nov: filter stopwords done. length: {len(list_nov)}")
#
# selected_comments_dec = random.sample(comments_dec, 1000)
list_dec = process(comments_dec)
print(f"Dec: filter stopwords done. length: {len(list_dec)}")

# tf_idf过滤
tf_idf_threshold = 0.2
length_threshold = 20
filtered_words_sep = tf_idf_filter(list_sep, tf_idf_threshold, length_threshold)
print("Sep: filter tf-idf done")
filtered_words_oct = tf_idf_filter(list_oct, tf_idf_threshold, length_threshold)
print("Oct: filter tf-idf done")
filtered_words_nov = tf_idf_filter(list_nov, tf_idf_threshold, length_threshold)
print("Nov: filter tf-idf done")
filtered_words_dec = tf_idf_filter(list_dec, tf_idf_threshold, length_threshold)
print("Dec: filter tf-idf done")

write(".\\output\\sep.txt", filtered_words_sep)
write(".\\output\\oct.txt", filtered_words_oct)
write(".\\output\\nov.txt", filtered_words_nov)
write(".\\output\\dec.txt", filtered_words_dec)
