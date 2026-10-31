# 导包
import jieba
import joblib
from scipy.sparse import hstack  # 把词级特征和字级特征拼在一起

# 1.加载打包好的向量器和模型
pipeline = joblib.load('../../model/ml/评论情感分类.pkl')
word_tfidf = pipeline['word_tfidf']   # 词级向量器
char_tfidf = pipeline['char_tfidf']   # 字级向量器
estimator = pipeline['estimator']     # 分类模型

# 2.标签映射：训练时统一做了 label + 1，预测结果要减回来看
label_dict = {0: '负面', 1: '中性', 2: '正面'}

# 3.把一条新评论转成和训练时完全一样的特征
def comment_feature(comment):
    # 3.1词级特征：先分词再用空格拼回字符串，然后用训练好的向量器转换
    comment_seg = ' '.join(jieba.lcut(comment))
    word_matrix = word_tfidf.transform([comment_seg])

    # 3.2字级特征：直接用原句
    char_matrix = char_tfidf.transform([comment])

    # 3.3横向拼接成和训练时同样的特征矩阵
    x = hstack([word_matrix, char_matrix]).tocsr()

    return x

# 4.预测一条评论的情感
def comment_predict(comment):
    # 4.1把评论转成特征
    x = comment_feature(comment)

    # 4.2模型预测
    y_pred = estimator.predict(x)[0]

    # 4.3打印结果
    print(f'评论：{comment}')
    print(f'预测情感：{label_dict[y_pred]}（{pipeline["model_name"]}模型，label={y_pred - 1}）\n')

# 5.测试
if __name__ == '__main__':
    comment_predict('这期视频的文案太好了，看几遍都不够')
    comment_predict('越来越失望了，完全没有以前的感觉')
    comment_predict('2026年来考古，有人吗')
