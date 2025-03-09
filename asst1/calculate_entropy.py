import os
import re
import math
import time
import jieba
import string
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

train_data_chinese_path = '../wiki_zh/train'
test_data_chinese_path = '../wiki_zh/test'
stopwords_chinese_path = '../cn_stopwords.txt'


def loaddata_chinese(data_path, mode):
    """
        加载中文数据
        :return:
    """
    data = ""

    #加载路径下所有内容，去除停用词后返回至data
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                filedata = f.read()
            data += filedata

    #按照模式进行分词
    split_words = []
    if mode == 'token':
        split_words = list(jieba.cut(data))
    elif mode == 'char':
        split_words = [ch for ch in data]

    # 保留所有的中文字符并去除分词中的停用词
    with open(stopwords_chinese_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    split_words = [word for word in split_words if re.match(r'^[\u4e00-\u9fff]+$', word) and word not in stopwords]

    return split_words


def loaddata_english_train(mode):
    """
        # 加载英文训练集数据
        # :return:
    """
    nltk.download('gutenberg')
    fileid_to_skip = 'austen-emma.txt'
    data = ""

    # 加载训练集中所有内容，此处将2以后的内容划为训练集
    for fileid in gutenberg.fileids():
        if fileid == fileid_to_skip:
            continue  # 跳过这个文件ID
        data += gutenberg.raw(fileid)

    # 按照模式进行分词
    split_words = []
    if mode == 'token':
        split_words = word_tokenize(data.lower())
    elif mode == 'char':
        split_words = list(data)

    #去除分词中的停用词
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    split_words = [word for word in split_words if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    split_words = [word for word in split_words if word not in stop_words]

    return split_words


def loaddata_english_test(mode):
    """
        加载英文测试集数据
        :return:
    """
    nltk.download('gutenberg')
    fileid = 'austen-emma.txt'

    # 加载测试集中所有内容，此处将1的内容划为测试集
    data = gutenberg.raw(fileid)

    # 按照模式进行分词
    split_words = []
    if mode == 'token':
        split_words = word_tokenize(data.lower())
    elif mode == 'char':
        split_words = list(data)

    # 去除分词中的停用词
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    split_words = [word for word in split_words if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    split_words = [word for word in split_words if word not in stop_words]

    return split_words

def get_tf(tf_dic, words):
    """
    获取一元词词频
    :return:一元词词频dic
    """
    for i in range(len(words)):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1


def get_bigram_tf(tf_dic, words):
    """
    获取二元词词频
    :return:二元词词频dic
    """
    for i in range(len(words)-1):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) + 1


def get_trigram_tf(tf_dic, words):
    """
    获取三元词词频
    :return:三元词词频dic
    """
    for i in range(len(words)-2):
        tf_dic[((words[i], words[i+1]), words[i+2])] = tf_dic.get(((words[i], words[i+1]), words[i+2]), 0) + 1


def calculate_unigram_entropy(tf_dic_train, tf_dic_test):
    """
    计算一元词的信息熵
    :return:
    """

    begin = time.time()
    words_num_train = sum([item[1] for item in tf_dic_train.items()])
    words_num_test = sum([item[1] for item in tf_dic_test.items()])
    entropy = 0
    for item in tf_dic_test.items():
        jp = item[1]/words_num_test
        item_f_intrain = tf_dic_train.get(item[0], 0)
        # 平滑处理
        if item_f_intrain == 0:
            item_f_intrain = 1
        cp = item_f_intrain / words_num_train
        if cp > 0:
            entropy += -jp * math.log(cp, 2)
    print("一元模型信息熵为：{:.6f} 比特/(词or字）".format(entropy))
    end = time.time()
    print("一元模型运行时间：{:.6f} s".format(end - begin))

    return ['unigram', round(entropy, 6)]


def calculate_bigram_entropy(tf_dic_train, bigram_tf_dic_train, bigram_tf_dic_test):
    """
    计算二元词的信息熵
    :return:
    """

    begin = time.time()
    bi_words_num_test = sum([item[1] for item in bigram_tf_dic_test.items()])
    entropy = 0
    for bi_item in bigram_tf_dic_test.items():
        jp = bi_item[1] / bi_words_num_test
        item_f_intrain = bigram_tf_dic_train.get(bi_item[0], 0)
        t = tf_dic_train.get(bi_item[0][0], 0)
        # 平滑处理
        if item_f_intrain == 0:
            item_f_intrain = 1
        if t == 0:
            t = 1
        cp = item_f_intrain / t
        if cp > 0:
            entropy += -jp * math.log(cp, 2)
    print("二元模型信息熵为：{:.6f} 比特/(词or字）".format(entropy))
    end = time.time()
    print("二元模型运行时间：{:.6f} s".format(end - begin))

    return ['bigram', round(entropy, 6)]


def calculate_trigram_entropy(bigram_tf_dic_train, trigram_tf_dic_train, trigram_tf_dic_test):
    """
    计算三元词的信息熵
    :return:
    """

    begin = time.time()
    tri_words_num_test = sum([item[1] for item in trigram_tf_dic_test.items()])
    entropy = 0
    for tri_item in trigram_tf_dic_test.items():
        jp = tri_item[1] / tri_words_num_test
        item_f_intrain = trigram_tf_dic_train.get(tri_item[0], 0)
        t = bigram_tf_dic_train.get(tri_item[0][0], 0)
        # 平滑处理
        if item_f_intrain == 0:
            item_f_intrain = 1
        if t == 0:
            t = 1
        cp = item_f_intrain / t
        if cp > 0:
            entropy += -jp * math.log(cp, 2)
    print("三元模型信息熵为：{:.6f} 比特/(词or字）".format(entropy))
    end = time.time()
    print("三元模型运行时间：{:.6f} s".format(end - begin))

    return ['trigram', round(entropy, 6)]

if __name__ == "__main__":

    mode_type = ['char', 'token']
    language_type = ['ch', 'en']

    for mode in mode_type:
        for language in language_type:

            tf_dic_train = {}
            bigram_tf_dic_train = {}
            trigram_tf_dic_train = {}
            tf_dic_test = {}
            bigram_tf_dic_test = {}
            trigram_tf_dic_test = {}

            if language == 'ch':

                data_train = loaddata_chinese(train_data_chinese_path, mode)
                data_test = loaddata_chinese(test_data_chinese_path, mode)
                print(f'中文信息熵计算结果({mode})：')

            if language == 'en':

                data_train = loaddata_english_train(mode)
                data_test = loaddata_english_test(mode)
                print(f'英文信息熵计算结果({mode})：')

            get_tf(tf_dic_train, data_train)
            get_bigram_tf(bigram_tf_dic_train, data_train)
            get_trigram_tf(trigram_tf_dic_train, data_train)
            get_tf(tf_dic_test, data_test)
            get_bigram_tf(bigram_tf_dic_test, data_test)
            get_trigram_tf(trigram_tf_dic_test, data_test)

            calculate_unigram_entropy(tf_dic_train, tf_dic_test)
            calculate_bigram_entropy(tf_dic_train, bigram_tf_dic_train, bigram_tf_dic_test)
            calculate_trigram_entropy(bigram_tf_dic_train, trigram_tf_dic_train, trigram_tf_dic_test)