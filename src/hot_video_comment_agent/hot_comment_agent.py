# 导包
import os
import jieba
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI

# 1.读取数据
comments = pd.read_csv('../../data/hot_comments_clean_processed_label.csv')
videos = pd.read_csv('../../data/hot_video_processed.csv')
comments['情感'] = comments['label'].map({-1: '负面', 0: '中性', 1: '正面'})

# 2.统计情感分布
def sentiment_stats():
    counts = comments['情感'].value_counts()
    total = len(comments)

    return {
        '总数': total,
        '负面': int(counts.get('负面', 0)),
        '中性': int(counts.get('中性', 0)),
        '正面': int(counts.get('正面', 0)),
        '负面占比': round(float(counts.get('负面', 0) / total), 4),
        '中性占比': round(float(counts.get('中性', 0) / total), 4),
        '正面占比': round(float(counts.get('正面', 0) / total), 4),
    }

# 3.按关键词搜评论
def search_comments(keyword, top_k=5):
    hit = comments[comments['评论内容'].str.contains(keyword, regex=False, na=False)]
    hit = hit.sort_values('点赞数', ascending=False).head(top_k)

    return [
        {
            '视频': bvid_to_title(row['bvid']),
            '评论内容': row['评论内容'][:100],
            '点赞数': int(row['点赞数']),
            '情感': row['情感'],
        }
        for _, row in hit.iterrows()
    ]

# 4.按指标给视频排名
def video_rank(field='播放', top_n=5):
    if field not in ['播放', '点赞', '硬币', '收藏', '分享', '评论数']:
        return {'错误': f'不支持的排序字段：{field}'}
    rank = videos.sort_values(field, ascending=False).head(top_n)

    return [
        {'标题': row['标题'], 'bvid': row['bvid'], 'up主': row['up主'], field: int(row[field])}
        for _, row in rank.iterrows()
    ]

# 5.查一个视频的指标和它下面的评论情感
def video_detail(title_keyword):
    hit = videos[videos['标题'].str.contains(title_keyword, regex=False, na=False)]
    if len(hit) == 0:
        return {'错误': f'没有找到标题包含“{title_keyword}”的视频'}
    video = hit.iloc[0]
    own = comments[comments['bvid'] == video['bvid']]

    return {
        '标题': video['标题'],
        'up主': video['up主'],
        '播放': int(video['播放']),
        '点赞': int(video['点赞']),
        '投币': int(video['硬币']),
        '评论样本数': len(own),
        '情感分布': own['情感'].value_counts().to_dict() if len(own) else {},
    }

# 6.判断一条新评论的情感
def predict_sentiment(text):
    pipeline = joblib.load('../../model/ml/评论情感分类.pkl')
    word_matrix = pipeline['word_tfidf'].transform([' '.join(jieba.lcut(text))])
    char_matrix = pipeline['char_tfidf'].transform([text])
    x = hstack([word_matrix, char_matrix]).tocsr()
    estimator = pipeline['estimator']
    y_pred = estimator.predict(x)[0]
    decision = estimator.decision_function(x)[0]
    exp_scores = np.exp(decision - decision.max())
    y_prob = exp_scores / exp_scores.sum()

    return {'评论': text, '情感': ['负面', '中性', '正面'][y_pred], '置信度': round(float(y_prob.max()), 4)}

# 7.bvid换成视频标题
def bvid_to_title(bvid):
    hit = videos[videos['bvid'] == bvid]

    return hit.iloc[0]['标题'] if len(hit) else bvid

# 8.把函数包成 LangChain工具，description是给模型看的用法说明
tools = [
    StructuredTool.from_function(
        func=sentiment_stats,
        name='sentiment_stats',
        description='统计全部评论中正面、中性、负面各有多少条以及占比。问“负面评论多不多”这类问题时用。',
    ),
    StructuredTool.from_function(
        func=search_comments,
        name='search_comments',
        description='按关键词搜索评论原文，返回点赞最高的几条。问“大家怎么评价某某”这类问题时用。',
    ),
    StructuredTool.from_function(
        func=video_rank,
        name='video_rank',
        description='按播放、点赞、硬币、收藏、分享、评论数给视频排名。问“哪个视频播放最高”这类问题时用。',
    ),
    StructuredTool.from_function(
        func=video_detail,
        name='video_detail',
        description='查某个视频的播放点赞投币，以及它下面评论的情感分布。需要视频标题的关键词。',
    ),
    StructuredTool.from_function(
        func=predict_sentiment,
        name='predict_sentiment',
        description='判断一条新评论是正面、中性还是负面。用户给一句评论让你判断情绪时用。',
    ),
]

# 9.配置大模型
def get_llm():
    # 9.1从环境变量读 Key
    api_key = os.environ['DASHSCOPE_API_KEY']
    base_url = 'https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1'
    model_name = 'deepseek-v4.1-flash'

    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name, temperature=0)

# 10.组装 Agent
def build_executor(llm):
    # 10.1工具调用
    prompt = ChatPromptTemplate.from_messages([
        ('system', '你是B站视频评论区数据分析助手。优先调用工具拿真实数据，不要自己编数字；工具报错就直接说明。'),
        ('human', '{input}'),
        MessagesPlaceholder('agent_scratchpad'),
    ])

    # 10.2模型每轮自己决定调哪个工具或者直接回答
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 10.3AgentExecutor负责循环：调工具、把结果还给模型、再决定下一步，最多 3 轮
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True,
    )
    return executor

# 11.提问并回答
def agent_answer(question):
    executor = build_executor(get_llm())
    result = executor.invoke({'input': question})
    return result['output']

# 12.打印一次问答
def agent_print(question):
    print(f'问题：{question}')
    print(f'回答：{agent_answer(question)}\n')


# 13.测试
if __name__ == '__main__':
    agent_print('哪个视频的播放最高')
    agent_print('负面评论有多少条')
    agent_print('二舅那个视频数据怎么样')
    agent_print('帮我判断这条评论的情感：这期视频太好看了，文案太顶了')
