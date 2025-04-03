import gensim

# 测试词对
if __name__ == '__main__':
    model = gensim.models.Word2Vec.load('model1.model')

    test_name = ['郭靖', '杨过', '段誉', '令狐冲', '张无忌']
    for name in test_name:
        print(name)
        for result in model.wv.similar_by_word(name, topn=10):
            print(result[0], '{:.6f}'.format(result[1]))
        print('----------------------')

    print(model.wv.doesnt_match("令狐冲 林平之 岳灵珊 岳不群 宁中则 陆大有 左冷禅".split()))