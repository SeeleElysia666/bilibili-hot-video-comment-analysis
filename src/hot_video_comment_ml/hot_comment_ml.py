import pandas as pd
import numpy as np
import jieba
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV  # 训练集和测试集的划分
from sklearn.metrics import classification_report     # 模型(分类)评估报告
from sklearn.model_selection import StratifiedKFold   # 分层K折交叉验证，类似于 网格搜索时 cv=折数
from sklearn.utils import class_weight
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack  # 把词级特征和字级特征拼在一起
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

# 1.读取人工标注后的数据
comments = pd.read_csv('../../data/hot_comments_clean_processed_label.csv')

# 2.数据预处理
def hot_comments_segmentation(comments):
    # 2.1对用户评论做jieba分词
    comments_list = [' '.join(jieba.lcut(comments)) for comments in comments['评论内容']]

    # 2.2加载停用词列表，即里面记录的词不需要参与切词
    with open('../../data/hit_stopwords.txt', 'r', encoding='utf-8') as f:
        # 一次读取所有行
        stopwords1 = f.readlines()
        # 删除最后'\n'
        stopwords1 = [stopword.strip() for stopword in stopwords1]
        # 对停用词去重
        stopwords_list1 = list(set(stopwords1))

    # 2.3自定义停用词表
    '''
    单使用哈工大停用词表，会缺少部分停用词
    '''
    stopwords_list2 = ['lex', '①①', '①②', '①③', '①④', '①⑤', '①⑥', '①⑦', '①⑧', '①⑨', '①ａ', '①ｂ', '①ｃ', '①ｄ', '①ｅ', '①ｆ','①ｇ', '①ｈ', '①ｉ', '①ｏ', '②①', '②②', '②③', '②④', '②⑤', '②⑥', '②⑦', '②⑧', '②⑩', '②ａ', '②ｂ', '②ｄ','②ｅ', '②ｆ', '②ｇ', '②ｈ', '②ｉ', '②ｊ', '③①', '③⑩', '③ａ', '③ｂ', '③ｃ', '③ｄ', '③ｅ', '③ｆ', '③ｇ', '③ｈ','④ａ', '④ｂ', '④ｃ', '④ｄ', '④ｅ', '⑤ａ', '⑤ｂ', '⑤ｄ', '⑤ｅ', '⑤ｆ', '１２', 'ｌｉ', 'ｚｘｆｉｔｌ']

    # 2.4拼接停用词表
    stopwords_list = stopwords_list1 + stopwords_list2

    return comments_list, stopwords_list

# 3.特征工程
def hot_comment_feature(comments_list, stopwords_list):
    # 3.1词级特征
    word_tfidf = TfidfVectorizer(stop_words=stopwords_list, ngram_range=(1, 2))
    word_matrix = word_tfidf.fit_transform(comments_list)

    # 3.2字级特征：char_wb在词边界内部取n-gram
    char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), sublinear_tf=True, min_df=2, max_df=0.95)
    char_matrix = char_tfidf.fit_transform(comments['评论内容'])

    # 3.3横向拼接成一个大矩阵
    x = hstack([word_matrix, char_matrix]).tocsr()

    # 3.4把两个向量器一起返回：新的评论要用同一套向量器转成特征才能拿来预测
    return x, word_tfidf, char_tfidf

# 4.训练模型
def hot_comment_model(x):
    # 4.1标签变成0,1,2
    y = comments['label'] + 1

    # 4.2划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # 4.3加入平衡权重，因为数据样本不均衡
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

    # 4.4创建模型对象
    estimators = {
        # TF-IDF是稀疏高维向量，用余弦距离；k太小会被噪声邻居带偏
        'KNN': KNeighborsClassifier(n_neighbors=10, metric='cosine'),
        # C太大会过拟合掉点
        '逻辑回归': LogisticRegression(C=1.0, class_weight='balanced'),
        # 增加树
        '随机森林': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1),
        # 多分类问题，使用多分类模型
        'XGBoost': xgb.XGBClassifier(n_estimators=300, objective='multi:softmax', random_state=42),
        # alpha 越小越不平滑，准确率和少数类召回都明显更好
        '朴素贝叶斯': MultinomialNB(alpha=0.1),
        # 文本分类用线性核
        'SVM': LinearSVC(C=0.3, class_weight='balanced'),
    }

    # 4.5模型训练
    result = {'names': [], 'accuracy': [], 'macro_f1': [], 'predict': {}}
    for name, estimator in estimators.items():
        if name in ['KNN', '逻辑回归', '随机森林', 'SVM']:
            estimator.fit(x_train, y_train)
        else:
            estimator.fit(x_train, y_train, sample_weight=classes_weights)

        # 4.6模型预测
        y_pred = estimator.predict(x_test)

        # 4.7模型评估
        result['names'].append(name)
        result['accuracy'].append(accuracy_score(y_test, y_pred))
        result['macro_f1'].append(f1_score(y_test, y_pred, average='macro'))
        result['predict'][name] = y_pred
        print(f'{name}模型的准确率为:{accuracy_score(y_test, y_pred)}\n{name}模型的分类评估报告为：\n{classification_report(y_test, y_pred)}')

        # 4.8保存模型
        joblib.dump(estimator, f'../../model/ml/{name}.pkl')

    print('所有模型保存成功')

    result['y_test'] = y_test
    return result

# 5.模型评测
def model_assessment(x):
    # 5.1标签变成0,1,2
    y = comments['label'] + 1

    # 5.2划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # 5.3加入平衡权重，因为数据样本不均衡
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

    # 5.4定义变量
    param_grids = {
        'KNN': {'n_neighbors': [5, 10, 20]},
        '逻辑回归': {'max_iter': [100, 300, 500], 'C': [0.1, 0.3, 1]},
        '随机森林': {'n_estimators': [100, 300, 500], 'max_depth': [5, 10, 20, None]},
        'XGBoost': {'n_estimators': [100, 300], 'max_depth': [5, 10], 'learning_rate': [0.1, 0.3]},
        '朴素贝叶斯': {'alpha': [0.01, 0.1, 1.0]},
        'SVM': {'C': [0.1, 0.3, 1]}
    }

    # 5.5创建分层采样对象
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # n_splits折数，shuffle是否打乱顺序

    # 5.6逐个加载并评测
    for name in ['KNN', '逻辑回归', '随机森林', 'XGBoost', '朴素贝叶斯', 'SVM']:
        estimator = joblib.load(f'../../model/ml/{name}.pkl')
        gs_estimator = GridSearchCV(estimator=estimator, param_grid=param_grids[name], cv=skf)
        if name in ['KNN', '逻辑回归', '随机森林', 'SVM']:
            gs_estimator.fit(x_train, y_train)
        else:
            gs_estimator.fit(x_train, y_train, sample_weight=classes_weights)

        # 5.7最优预测
        y_pred = gs_estimator.predict(x_test)

        # 5.8模型评估
        print(f'{name}模型最优评分：{gs_estimator.best_score_}\n{name}模型最优估计器组合：\n{gs_estimator.best_estimator_}\n{name}模型的最优分类评估报告为：\n{classification_report(y_test, y_pred)}')

        # 5.9保存模型
        joblib.dump(gs_estimator.best_estimator_, f'../../model/ml/{name}_best.pkl')

    print('所有最优模型保存成功')


# 6.绘制各模型准确率对比柱状图
def draw_accuracy_bar(result):
    # 6.1按准确率从高到低排序，方便看图
    order = np.argsort(result['accuracy'])[::-1]
    names = [result['names'][i] for i in order]
    accuracy = [result['accuracy'][i] for i in order]
    macro_f1 = [result['macro_f1'][i] for i in order]

    # 6.2准确率和macro-F1并排画，因为样本不均衡时只看准确率会片面
    x_pos = np.arange(len(names))
    plt.figure(figsize=(12, 7))
    plt.bar(x_pos - 0.2, accuracy, width=0.4, label='准确率')
    plt.bar(x_pos + 0.2, macro_f1, width=0.4, label='macro-F1')

    # 6.3在柱子上标数值
    for i in range(len(names)):
        plt.text(x_pos[i] - 0.2, accuracy[i] + 0.01, f'{accuracy[i]:.4f}', ha='center', fontsize=10)
        plt.text(x_pos[i] + 0.2, macro_f1[i] + 0.01, f'{macro_f1[i]:.4f}', ha='center', fontsize=10)

    plt.xticks(x_pos, names)
    plt.ylim(0, 1)
    plt.xlabel('模型')
    plt.ylabel('得分')
    plt.title('6个模型在测试集上的准确率与macro-F1对比（测试集1813条）')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../data/fig/ml/各模型准确率对比柱状图.png')
    plt.show()

# 7.绘制逻辑回归的混淆矩阵
def draw_confusion_matrix(result):
    # 6.1混淆矩阵：行是真实标签，列是预测标签
    cm = confusion_matrix(result['y_test'], result['predict']['逻辑回归'])
    labels = ['负面', '中性', '正面']

    # 7.2画热力图
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('逻辑回归混淆矩阵（测试集1813条）')
    plt.tight_layout()
    plt.savefig('../../data/fig/ml/逻辑回归混淆矩阵.png')
    plt.show()


# 8.模型打包
def hot_comment_pipeline(word_tfidf, char_tfidf):
    # 8.1取准确率最高的逻辑回归作为最终模型
    estimator = joblib.load('../../model/ml/逻辑回归.pkl')

    # 8.2把两个向量器和模型打包成一份文件：新的评论先用向量器转成特征，再交给模型预测
    pipeline = {
        'model_name': '逻辑回归',
        'word_tfidf': word_tfidf,
        'char_tfidf': char_tfidf,
        'estimator': estimator
    }

    # 8.3保存打包文件
    joblib.dump(pipeline, '../../model/ml/评论情感分类.pkl')
    print('模型打包完成')


# 9.测试
if __name__ == '__main__':
    comments_list, stopwords_list = hot_comments_segmentation(comments)
    x, word_tfidf, char_tfidf = hot_comment_feature(comments_list, stopwords_list)
    result = hot_comment_model(x)
    model_assessment(x)
    draw_accuracy_bar(result)
    draw_confusion_matrix(result)
    hot_comment_pipeline(word_tfidf, char_tfidf)