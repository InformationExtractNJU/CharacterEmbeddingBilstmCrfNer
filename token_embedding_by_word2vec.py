from gensim.models import word2vec

# word2vec.Text8Corpus('NewsCar_new_after_process/1/2.txt')

# 加载分句后的文件
reader = open('../train_data/sentences.txt',encoding = 'utf-8-sig')
train_data = reader.readlines()

print (train_data[0])

tokens = []
for i in range(len(train_data)):
    if(i%2 == 0):
        sen = train_data[i].split(" ")
        if(sen[len(sen)-1]=='\n'):
            sen.pop()
        tokens.append(sen)
#
print (tokens)
model = word2vec.Word2Vec(tokens, size=256, min_count=1)

# size 表示向量维度 min_count表示最小出现次数
# model = word2vec.Word2Vec.load("word2vec_model/word2vec.model")
# #
# # # 计算和车最相似的5个字
# x=model.most_similar("月",topn=5)
# print (x)
# #
# simliar=model.similarity('汽','车')
# print (simliar)
# # # 输出'汽车'的词向量
# # print(model[['汽','车']])
#
# two_dim=model[['汽','车']]
# res=[]
# for word in two_dim:
#     word_vec=[]
#     for j in word:
#         dim_vec=[]
#         dim_vec.append(j)
#         word_vec.append(dim_vec)
#     res.append(word_vec)
#
# print (res)

# vocab=model.wv.vocab
# for word in vocab:
#     print (word)
#
# # 保存模型
model.save("word2vec_model/word2vec.model")
# 对应的加载方式
# model_2 = word2vec.Word2Vec.load("text8.model")



