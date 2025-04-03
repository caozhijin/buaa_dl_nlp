import re
import jieba
from gensim.models import Word2Vec

data_path = '../data/'
inf_path = '../data/inf.txt'
stopwords_path = '../cn_stopwords.txt'

def get_file_data(file_path):
    """
    获取file_path下文件内容并进行预处理
    """
    lines = []

    # unuseful items filter
    with open(file_path, 'r', encoding='ANSI') as f:
        for line in f:  # 分别对每段分词
            line = line.replace('本书来自www.cr173.com免费txt小说下载站', '')
            line = line.replace('更多更新免费电子书请关注www.cr173.com', '')
            split_words = list(jieba.cut(line))  # 结巴分词 精确模式
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                stopwords = [line.strip() for line in f.readlines()]

            split_words = [word for word in split_words if re.match(r'^[\u4e00-\u9fff]+$', word) and word not in stopwords]
            if len(split_words) > 0:
                lines.append(split_words)

        # print(lines[0:5])
    return lines


def get_all_data():
    """
        加载所有txt文件，返回all_data
    """
    all_data = []

    with open(inf_path, 'r') as f:
        txt_list = f.readline().split(',')
        for index, file in enumerate(txt_list):
            file_path = data_path + file + '.txt'
            all_data += get_file_data(file_path)

    return all_data


if __name__ == '__main__':
    all_data = get_all_data()
    model = Word2Vec(sentences=all_data, min_count=10, window=5,vector_size=200, hs=1, sg=0,epochs=500)
    model.save('model1.model')