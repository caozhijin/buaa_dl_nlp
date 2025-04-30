import re
import jieba
import torch

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

def build_vocab(tokens, min_freq=1):
    freq = {}
    for word in tokens:
        freq[word] = freq.get(word, 0) + 1
    # 初始化特殊标记
    vocab = {"<unk>": 0, "<pad>": 1, "<eos>": 2}
    for word, count in freq.items():
        if word not in vocab and count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def numericalize(tokens, vocab):
    """
    将文本中每个词转换成对应的索引。如果词不在词汇表中，则返回 <unk> 对应的索引
    """
    return [vocab.get(word, vocab["<unk>"]) for word in tokens]

class LanguageModelDataset():
    def __init__(self, token_ids, seq_length):
        """
        初始化数据集：
        - token_ids: 数值化后的词索引列表
        - seq_length: 每个训练样本的长度
        每个样本由输入序列 [w0, w1, ..., w_{L-1}] 和目标序列 [w1, w2, ..., w_L] 组成
        """
        self.token_ids = token_ids
        self.seq_length = seq_length

    def __len__(self):
        return len(self.token_ids) - self.seq_length

    def __getitem__(self, idx):
        x = self.token_ids[idx: idx + self.seq_length]
        y = self.token_ids[idx + 1: idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


if __name__ == '__main__':
    file = data_path + '笑傲江湖.txt'
    d = get_file_data(file)
    vocab = build_vocab( d, min_freq=1)
    token_ids = numericalize( d, vocab)
    seq_length = 50
    dataset = LanguageModelDataset(token_ids, seq_length)

    # x, y = dataset[0]
    # print("Input sequence:", x)
    # print("Target sequence:", y)