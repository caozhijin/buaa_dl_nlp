import pandas as pd
import jieba
import lda
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import cross_val_score

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如使用黑体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

inf_path = '../data/inf.txt'
data_path = '../data/'
stopwords_path = '../cn_stopwords.txt'

def get_file_docs(file_path, mode, num, length):
    """
    获取file_path文件对应的段落，段落个数为num，段落长度为length
    返回多个段落的列表
    """
    corpus = ""

    # unuseful items filter
    with open(file_path, 'r', encoding='ANSI') as f:
        corpus = f.read()
        corpus = corpus.replace('本书来自www.cr173.com免费txt小说下载站', '')
        corpus = corpus.replace('更多更新免费电子书请关注www.cr173.com', '')
        f.close()

    split_words = []
    if mode == 'word':
        split_words = list(jieba.cut(corpus))
    elif mode == 'char':
        split_words = [ch for ch in corpus]

    # 保留所有的中文字符并去除分词中的停用词
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    split_words = [word for word in split_words if re.match(r'^[\u4e00-\u9fff]+$', word) and word not in stopwords]

    # 随机提取num个大小为length的段
    segments = []

    for i in range(num):
        start_index = i*int(len(split_words)/num)
        segment = split_words[start_index:start_index + length]
        segments.append(segment)

    return segments


def get_all_docs(mode, length):
    """
        加载docs
        返回有1000个段落列表的列表all_docs
    """
    all_docs = []

    with open(inf_path, 'r') as f:
        txt_list = f.readline().split(',')
        for index, file in enumerate(txt_list):
            file_path = data_path + file + '.txt'
            num = 62 if index < 8 else 63
            all_docs += get_file_docs(file_path, mode, num, length)
        f.close()

    return all_docs

def get_docs_words(all_docs):
    """
    创建一个二维字典，每行对应 all_docs 中的一个段落，每列对应所有单词的出现频率。
    返回一个 Pandas DataFrame 和行列索引与编号的映射。
    """
    all_words = set()
    for segment in all_docs:
        for word in segment:
            all_words.add(word)

    # 初始化二维字典
    docs_word_freq = {segment_index: {word: 0 for word in all_words} for segment_index in range(len(all_docs))}

    for segment_index, segment in enumerate(all_docs):
        for word in segment:
            docs_word_freq[segment_index][word] += 1

    # 将二维字典转换为 Pandas DataFrame
    df = pd.DataFrame(docs_word_freq).T  # 转置以使每行对应一个段落

    # 获取行列索引与编号的映射
    with open(inf_path, 'r') as f:
        txt_list = f.readline().split(',')
        row_index_to_id = {i: file_name for i, file_name in enumerate(txt_list)}
    col_index_to_word = {i: word for i, word in enumerate(all_words)}

    return df, row_index_to_id, col_index_to_word

if __name__ == "__main__":

    mode_type = ['word']
    token_length_type = [3000]
    num_topic_T_type = [5, 10, 15, 20, 25]


    for mode in mode_type:
        for token_length in token_length_type:

            all_docs = get_all_docs(mode, token_length)
            df, row_index_to_id, col_index_to_word = get_docs_words(all_docs)

            for num_topic_T in num_topic_T_type:
                model = lda.LDA(n_topics=num_topic_T, n_iter=1500, random_state=1)
                model.fit(df.values)
                doc_topic_matrix = model.doc_topic_

                # 生成 labels 列表
                labels = []
                for i in range(16):
                    num = 62 if i < 8 else 63
                    labels.extend([i] * num)

                # 创建 SVM 分类器
                clf = svm.SVC(kernel='linear')

                # 进行十次交叉验证
                scores = cross_val_score(clf, doc_topic_matrix, labels, cv=10)

                result = f"Mode: {mode}, Token Length: {token_length}, Num Topics: {num_topic_T}\n"
                result += f"Cross-validation scores: {scores}\n"
                result += f"Average cross-validation score: {scores.mean()}\n"

                # 将结果写入文件
                with open('output.txt', 'a', encoding='utf-8') as file:
                    file.write(result)

                # 绘制热图
                # plt.figure(figsize=(10, 8))
                # sns.heatmap(doc_topic_matrix[:10], annot=True, cmap='viridis',xticklabels=[f'Topic_{i + 1}' for i in range(num_topic_T)],)
                # plt.title('文档 - 主题分布热图')
                # plt.xlabel('主题')
                # plt.ylabel('文档')
                # plt.show()
