# 导包
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer # 把评论内容转成词频矩阵
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.sparse import hstack  # 把词级特征和字级特征拼在一起
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

# 1.读取文件
comments = pd.read_csv('../../data/hot_comments_clean_processed.csv')

# 2.加载停用词列表，即里面记录的词不需要参与切词
with open('../../data/hit_stopwords.txt', 'r', encoding='utf-8') as f:
    # 一次读取所有行
    stopwords1 = f.readlines()
    # 删除最后'\n'
    stopwords1 = [stopword.strip() for stopword in stopwords1]
    # 对停用词去重
    stopwords_list1 = list(set(stopwords1))

# 2.1自定义停用词表
'''
单使用哈工大停用词表，会缺少部分停用词
'''
stopwords_list2 = ['lex', '①①', '①②', '①③', '①④', '①⑤', '①⑥', '①⑦', '①⑧', '①⑨', '①ａ', '①ｂ', '①ｃ', '①ｄ', '①ｅ', '①ｆ', '①ｇ', '①ｈ', '①ｉ', '①ｏ', '②①', '②②', '②③', '②④', '②⑤', '②⑥', '②⑦', '②⑧', '②⑩', '②ａ', '②ｂ', '②ｄ', '②ｅ', '②ｆ', '②ｇ', '②ｈ', '②ｉ', '②ｊ', '③①', '③⑩', '③ａ', '③ｂ', '③ｃ', '③ｄ', '③ｅ', '③ｆ', '③ｇ', '③ｈ', '④ａ', '④ｂ', '④ｃ', '④ｄ', '④ｅ', '⑤ａ', '⑤ｂ', '⑤ｄ', '⑤ｅ', '⑤ｆ', '１２', 'ｌｉ', 'ｚｘｆｉｔｌ']

# 2.2拼接停用词表
stopwords_list = stopwords_list1 + stopwords_list2

# 3.绘制词云图
def hot_comments_wordcloud(comments):
    # 3.1对用户评论做jieba分词
    comments_list = comments['评论内容'].apply(lambda x: jieba.lcut(x))
    # 过滤停用词，同时去掉 jieba 切出来的空格
    comments_list = comments_list.apply(lambda x: [i for i in x if i.strip() and i not in stopwords_list])

    # 3.2统计词频
    hot_word = [word for words in comments_list for word in words]
    hot_word_count = pd.Series(hot_word).value_counts()
    print(f'热门评论前10高频词：\n{hot_word_count.head(10)}')

    # 3.3绘图
    plt.figure(figsize=(12, 8))
    wordcloud = WordCloud(
        font_path='simhei.ttf',
        width=800,
        height=600,
        background_color='white',
        max_words=300,
        colormap='viridis',
        contour_width=1,
        contour_color='steelblue'
    ).generate_from_frequencies(hot_word_count)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout()
    plt.savefig('../../data/fig/cluster/热门评论词云图.png')
    plt.show()

# 4.寻找最优k值
def cluster_k(comments):
    # 4.1对用户评论做jieba分词
    comments_list = [' '.join(jieba.lcut(comments)) for comments in comments['评论内容']]

    # 4.2创建向量化对象，从评论切词列表中删除停用词，并统计词频
    # 4.2.1词级特征
    word_tfidf = TfidfVectorizer(stop_words=stopwords_list, ngram_range=(1, 2))
    word_matrix = word_tfidf.fit_transform(comments_list)

    # 4.2.2字级特征：char_wb在词边界内部取n-gram
    char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), sublinear_tf=True, min_df=2, max_df=0.95)
    char_matrix = char_tfidf.fit_transform(comments['评论内容'])

    # 4.2.2SVD降维
    svd = TruncatedSVD(n_components=100, random_state=42) # 降到100维
    svd_matrix = svd.fit_transform(hstack([word_matrix, char_matrix]).tocsr())

    # 4.2定义列表，记录不同K值评估结果
    SSE_list = []  # 只考虑簇内，越小越好
    SC_list = []  # 考虑簇内和簇间，越大越好
    CH_list = [] # 考虑簇内的内聚程度、簇外的离散程度，类别之间的距离平方和越大越好。聚类的种类数越少越好

    # 4.3归一化：把每条样本压成单位长度再聚类
    x = normalize(svd_matrix)

    # 4.4定义for循环，测试不同K值评估效果
    for k in range(2, 10):
        # 4.4.1创建模型对象
        estimator = KMeans(n_clusters=k, random_state=42, n_init=10)

        # 4.4.2模型训练
        estimator.fit(x)

        # 4.4.3模型预测
        y_pred = estimator.predict(x)

        # 4.4.4添加评分
        SSE_list.append(estimator.inertia_)
        SC_list.append(silhouette_score(x, y_pred))
        CH_list.append(calinski_harabasz_score(x, y_pred))

    # 4.绘制折线图
    plt.figure(figsize=(20, 10))
    plt.plot(range(2, 10), SSE_list, label='SSE')
    plt.xlabel('聚类数量 k')
    plt.ylabel('SSE（簇内平方和）')
    plt.title('SSE肘部法则曲线')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../data/fig/cluster/SSE肘部法则曲线.png')
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(range(2, 10), SC_list, label='SC')
    plt.xlabel('聚类数量 k')
    plt.ylabel('轮廓系数 SC')
    plt.title('轮廓系数SC随k变化曲线')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../data/fig/cluster/轮廓系数SC随k变化曲线.png')
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(range(2, 10), CH_list, label='CH')
    plt.xlabel('聚类数量 k')
    plt.ylabel('CH指数')
    plt.title('CH指数随k变化曲线')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../data/fig/cluster/CH指数随k变化曲线.png')
    plt.show()

    return x

# 5.评论聚类
def comment_clustering(x):
    # 5.1训练模型
    estimator = KMeans(n_clusters=5, random_state=42, n_init=10)
    estimator.fit(x)

    # 5.2模型预测
    y_pred = estimator.predict(x)
    comments['cluster_label'] = y_pred

    # 5.3查看5个簇的评论关键词
    all_words = [w for txt in comments['评论内容'] for w in jieba.lcut(txt) if w not in stopwords_list and len(w) > 1]
    common_words = set(pd.Series(all_words).value_counts().head(50).index)

    for i in range(5):
        print(f'\n簇{i}前10热门词')
        hot_words = []
        for txt in comments[comments.cluster_label == i]['评论内容']:
            hot_words += [w for w in jieba.lcut(txt) if w not in stopwords_list and w not in common_words and len(w) > 1]
        print(pd.Series(hot_words).value_counts().head(10))

    # 5.4绘制t-SNE散点图5个簇样本点
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
    x_tsne = tsne.fit_transform(x)

    sc_list = []
    labels = []
    for k in range(5):
        sc = plt.scatter(x_tsne[y_pred == k, 0], x_tsne[y_pred == k, 1], alpha=0.6)
        sc_list.append(sc)
        labels.append(f'簇{k}')

    # 5.5绘制每个簇的质心：取该簇在t-SNE图上的坐标均值
    center_x = [x_tsne[y_pred == k, 0].mean() for k in range(5)]
    center_y = [x_tsne[y_pred == k, 1].mean() for k in range(5)]
    center = plt.scatter(center_x, center_y)
    sc_list.append(center)
    labels.append('质心')

    # 5.6设置标题
    plt.title('评论聚类散点图（t-SNE降维）')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')
    plt.legend(sc_list,labels, loc='best')
    plt.tight_layout()
    plt.savefig('../../data/fig/cluster/评论聚类散点图_tSNE.png')
    plt.show()


# 6.测试
if __name__ == '__main__':
    hot_comments_wordcloud(comments)
    x = cluster_k(comments)
    comment_clustering(x,)