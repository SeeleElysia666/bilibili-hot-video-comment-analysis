# 导包
import os
import pickle
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 1.建索引
def build_index():
    # 1.1读评论表和视频表
    index_dir = '../../model/rag'
    comments = pd.read_csv('../../data/hot_comments_clean_processed_label.csv')
    videos = pd.read_csv('../../data/hot_video_processed.csv')
    video_dict = dict(zip(videos['bvid'], videos['标题']))
    label_dict = {-1: '负面', 0: '中性', 1: '正面'}

    # 1.2字级切分，中文不用分词
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    matrix = vectorizer.fit_transform(comments['评论内容'])

    # 1.3整理每条评论的信息
    docs = [
        {
            'bvid': row['bvid'],
            '视频标题': video_dict.get(row['bvid'], '未知视频'),
            '评论内容': row['评论内容'],
            '点赞数': int(row['点赞数']),
            'label': int(row['label']),
            '情感': label_dict[int(row['label'])],
        }
        for _, row in comments.iterrows()
    ]

    # 1.4保存向量器、矩阵、评论信息
    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(index_dir, 'matrix.pkl'), 'wb') as f:
        pickle.dump(matrix, f)
    with open(os.path.join(index_dir, 'docs.pkl'), 'wb') as f:
        pickle.dump(docs, f)

    print(f'索引建立完成：{len(docs)} 条评论，特征维度 {matrix.shape[1]}')

# 2.加载索引
def load_index():
    index_dir = '../../model/rag'

    # 2.1没有索引就先建
    if not os.path.exists(os.path.join(index_dir, 'vectorizer.pkl')):
        build_index()

    with open(os.path.join(index_dir, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(index_dir, 'matrix.pkl'), 'rb') as f:
        matrix = pickle.load(f)
    with open(os.path.join(index_dir, 'docs.pkl'), 'rb') as f:
        docs = pickle.load(f)

    return vectorizer, matrix, docs

# 3.挑出问题里的少见词
def pick_rare_words(question, docs, max_df=60):
    # 3.1分词，去掉单字
    words = [word for word in jieba.lcut(question) if len(word) >= 2]

    # 3.2只保留出现在少量评论里的词
    rare_words = []
    for word in words:
        df = sum(1 for doc in docs if word in doc['评论内容'])
        if 0 < df <= max_df:
            rare_words.append(word)

    return rare_words

# 4.检索
def retrieve(question, vectorizer, matrix, docs, top_k=5):
    # 4.1问题转成向量
    question_vec = vectorizer.transform([question])

    # 4.2有少见词就只在候选评论里检索
    rare_words = pick_rare_words(question, docs)
    if rare_words:
        candidate_index = [i for i, doc in enumerate(docs) if any(word in doc['评论内容'] for word in rare_words)]
        print(f'命中稀有词：{rare_words}，候选评论 {len(candidate_index)} 条')
    else:
        candidate_index = list(range(len(docs)))
        print('没有命中稀有词，在全库检索')

    # 4.3算余弦相似度
    scores = cosine_similarity(question_vec, matrix[candidate_index])[0]

    # 4.4取相似度最高的前 k 条
    order = scores.argsort()[::-1][:top_k]

    # 4.5包成Document，正文放page_content，其余放metadata
    results = []
    for position in order:
        one = docs[candidate_index[position]]
        results.append(Document(
            page_content=one['评论内容'],
            metadata={
                '视频标题': one['视频标题'],
                '情感': one['情感'],
                '点赞数': one['点赞数'],
                '相似度': round(float(scores[position]), 4),
            }
        ))

    return results

# 5.把检索结果拼成一段材料
def format_docs(docs):
    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        lines.append(f'{i}. 【{meta["视频标题"]}｜{meta["情感"]}｜点赞{meta["点赞数"]}】{doc.page_content[:120]}')

    return '\n'.join(lines)

# 6.配置大模型
def get_llm():
    # 6.1从环境变量读Key
    api_key = os.environ['DASHSCOPE_API_KEY']
    base_url = 'https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1'
    model_name = 'deepseek-v4.1-flash'

    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name, temperature=0)

# 7.用LCEL把检索、提示词、模型串成一条链
def build_chain(llm, top_k=5):
    # 7.1检索器
    vectorizer, matrix, docs = load_index()
    retriever = RunnableLambda(lambda question: retrieve(question, vectorizer, matrix, docs, top_k))

    # 7.2提示词
    prompt = ChatPromptTemplate.from_messages([
        ('system', '你是B站评论区分析助手。只能根据用户给的评论材料回答，材料里没有的就直说不知道，不要编造。'),
        ('human', '下面是检索到的评论：\n{context}\n\n请根据这些评论回答问题：{question}'),
    ])

    # 7.3竖线表示把上一步的结果交给下一步
    chain = (
        {
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# 8.提问并回答
def rag_answer(question, top_k=5):
    chain = build_chain(get_llm(), top_k)

    return chain.invoke(question)

# 9.打印一次问答
def rag_print(question):
    print(f'\n问题：{question}')
    print(f'回答：{rag_answer(question)}')

# 10.测试
if __name__ == '__main__':
    load_index()
    rag_print('大家对二舅这个视频是什么感受')
    rag_print('新宝岛为什么这么火')
    rag_print('有没有人吐槽弹幕礼仪')
