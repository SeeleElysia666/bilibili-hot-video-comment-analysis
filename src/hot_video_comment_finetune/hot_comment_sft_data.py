# 导包
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1.包装一条训练样本
def build_one_sample(comment, label):
    # 1.1标签换成中文
    label_dict = {-1: '负面', 0: '中性', 1: '正面'}

    # 1.2任务指令
    system_prompt = '你是B站评论区情感分析助手，只根据评论内容判断情感倾向，只输出 负面、中性、正面 三个词中的一个，不要解释。'

    # 1.3chat 格式
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f'请判断下面这条B站评论的情感倾向：{comment}'},
        {'role': 'assistant', 'content': label_dict[label]},
    ]

    return {'messages': messages}

# 2.整表转样本
def build_all_samples(comments):
    # 2.1一行一条样本
    samples = [build_one_sample(row['评论内容'], row['label']) for _, row in comments.iterrows()]

    # 2.2打印样本量
    answer_list = [sample['messages'][2]['content'] for sample in samples]
    print(f'样本总数：{len(samples)}')
    print(f'标签分布：{pd.Series(answer_list).value_counts().to_dict()}')

    return samples

# 3.划分数据
def split_samples(comments):
    # 3.1按 8:2 切
    train_all_df, test_df = train_test_split(comments, test_size=0.2, random_state=42, stratify=comments['label'])

    # 3.2再切 10% 验证集
    train_df, valid_df = train_test_split(train_all_df, test_size=0.1, random_state=42, stratify=train_all_df['label'])

    print(f'训练集 {len(train_df)} 条，验证集 {len(valid_df)} 条，测试集 {len(test_df)} 条')

    return train_df, valid_df, test_df

# 4.写 jsonl文件
def save_jsonl(samples, path):
    # 4.1建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 4.2一行一个 json
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f'保存成功：{path}，{len(samples)} 条')

# 5.生成数据文件
def build_sft_data():
    # 5.1读取标注数据
    comments = pd.read_csv('../../data/hot_comments_clean_processed_label.csv')

    # 5.2切分数据
    train_df, valid_df, test_df = split_samples(comments)

    # 5.3转样本并保存
    save_jsonl(build_all_samples(train_df), '../../data/finetune/sft_train.jsonl')
    save_jsonl(build_all_samples(valid_df), '../../data/finetune/sft_valid.jsonl')
    save_jsonl(build_all_samples(test_df), '../../data/finetune/sft_test.jsonl')

    # 5.4测试集存 csv
    test_df.to_csv('../../data/finetune/sft_test.csv', index=False, encoding='utf-8-sig')
    print('保存成功：../../data/finetune/sft_test.csv')

# 6.测试
if __name__ == '__main__':
    build_sft_data()
