import random
import re
import jieba
import numpy as np

data_path = '../data/'
inf_path = '../data/inf.txt'
stopwords_path = '../cn_stopwords.txt'

def get_file_data(file_path):
    """
    获取file_path下文件内容并进行预处理
    """
    file_data = []

    # unuseful items filter
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:  # 分别对每段分词
            line = line.replace('本书来自www.cr173.com免费txt小说下载站', '')
            line = line.replace('更多更新免费电子书请关注www.cr173.com', '')
            split_words = list(jieba.cut(line))  # 结巴分词 精确模式
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                stopwords = [line.strip() for line in f.readlines()]
            split_words = [word for word in split_words if
                           re.match(r'^[\u4e00-\u9fff]+$', word) or word in stopwords]
            if len(split_words) > 0:
                file_data += split_words

        # print(file_data[0:50])
    return file_data

def get_all_data():
    """
        加载所有txt文件，返回all_data
    """
    all_data = []

    with open(inf_path, 'r', encoding='utf-8') as f:
        txt_list = f.readline().split(',')
        for index, file in enumerate(txt_list):
            file_path = data_path + file + '.txt'
            all_data += get_file_data(file_path)

    return all_data

def get_dataset(data):
    """
    :param data: 分词结果
    :return: 落库，段落对应的下一个词，词库和词库索引
    """
    max_len = 50
    step = 10
    sentences = []
    next_tokens = []

    tokens = list(set(data))
    tokens_indices = {token: tokens.index(token) for token in tokens}
    print('Unique tokens:', len(tokens))

    for i in range(0, len(data) - max_len, step):
        sentences.append(
            list(map(lambda t: tokens_indices[t], data[i: i + max_len])))
        next_tokens.append(tokens_indices[data[i + max_len]])
    print('Number of sequences:', len(sentences))

    print('Vectorization...')
    next_tokens_one_hot = []
    for i in next_tokens:
        y = np.zeros((len(tokens),))
        y[i] = 1
        next_tokens_one_hot.append(y)
    # print(sentences[0], next_tokens_one_hot[0])
    print(len(sentences), len(next_tokens_one_hot))
    return sentences, next_tokens_one_hot, tokens, tokens_indices

if __name__ == '__main__':
    all_data = get_all_data()
    _x, _y, _tokens, _tokens_indices = get_dataset(all_data)
