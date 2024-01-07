import csv
import random


def write(filepath, datalist):
    file = open(filepath, 'w', encoding='UTF-8')
    file.close()
    file = open(filepath, 'a', encoding='UTF-8')
    for item in datalist:
        file.write(item.replace("\n", ""))
        file.write("\n")
    file.close()


with open('reddit_opinion_PSE_ISR.csv', 'r', encoding='UTF-8') as f:
    # 创建一个csv.reader对象，指定分隔符为逗号
    reader = csv.reader(f, delimiter=',')

    # 遍历reader对象
    count = 0   # 数据条数
    comments_sep = []
    comments_oct = []
    comments_nov = []
    comments_dec = []
    for row in reader:
        if row[4][5:7] == '09':
            comments_sep.append(row[2])
        elif row[4][5:7] == '10':
            comments_oct.append(row[2])
        elif row[4][5:7] == '11':
            comments_nov.append(row[2])
        elif row[4][5:7] == '12':
            comments_dec.append(row[2])
        count += 1
        print(count)

    print(len(comments_sep))
    print(len(comments_oct))
    print(len(comments_nov))
    print(len(comments_dec))
    '''
    # 随机取1000条数据
    selected_comments_sep = random.sample(comments_sep, 1000)
    selected_comments_oct = random.sample(comments_oct, 1000)
    selected_comments_nov = random.sample(comments_nov, 1000)
    selected_comments_dec = random.sample(comments_dec, 1000)
    '''
    write(".\\input\\sep_comments.txt", comments_sep)
    write(".\\input\\oct_comments.txt", comments_oct)
    write(".\\input\\nov_comments.txt", comments_nov)
    write(".\\input\\dec_comments.txt", comments_dec)
