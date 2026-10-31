# 导包
import pandas as pd
import torch
import jieba
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1.加载模型
def get_models():
    # 1.1基座和适配器路径
    base_model_dir = '../../model/finetune/base_model/Qwen2.5-0.5B-Instruct'
    lora_dir = '../../model/finetune/lora_adapter'

    # 1.2加载基座
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 1.3生成时左边补齐
    tokenizer.padding_side = 'left'

    base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.float16, device_map='auto')
    base_model.eval()

    # 1.4加载适配器
    # 适配器贴回基座后，测基座要 disable_adapter()
    lora_model = PeftModel.from_pretrained(base_model, lora_dir)
    lora_model.eval()
    print('模型加载完成')

    return tokenizer, lora_model

# 2.标签词转 token 编号
def get_label_ids(tokenizer):
    # 每个词取第一个 token
    label_ids = {}
    for word in ['负面', '中性', '正面']:
        label_ids[word] = tokenizer.encode(word, add_special_tokens=False)[0]
    print(f'三个标签对应的 token：{label_ids}')

    return label_ids

# 3.拼提示词
def build_prompts(texts, tokenizer):
    # 3.1指令和训练时一致
    system_prompt = '你是B站评论区情感分析助手，只根据评论内容判断情感倾向，只输出 负面、中性、正面 三个词中的一个，不要解释。'

    prompts = []
    for text in texts:
        prompt = tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'请判断下面这条B站评论的情感倾向：{text}'},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    return prompts

# 4.用打分来预测
def model_score_predict(texts, tokenizer, label_ids, model, batch_size=32, name='模型'):
    results = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        prompts = build_prompts(batch, tokenizer)

        # 4.1取最后一个位置
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # 4.2比较三个标签词的分数
        last_logits = outputs.logits[:, -1, :].float()
        for row in last_logits:
            score_dict = {word: row[label_ids[word]].item() for word in ['负面', '中性', '正面']}
            results.append(max(score_dict, key=score_dict.get))

        print(f'{name}已完成 {min(start + batch_size, len(texts))}/{len(texts)} 条')

    return results

# 5.生成一个词
def model_generate_predict(texts, tokenizer, model, batch_size=8, name='模型'):
    results = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        prompts = build_prompts(batch, tokenizer)

        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        for i in range(len(batch)):
            gen_ids = outputs[i][inputs['input_ids'].shape[1]:]
            answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            hit = [word for word in ['负面', '中性', '正面'] if word in answer]
            results.append(hit[0] if hit else '无法判断')

        print(f'{name}已完成 {min(start + batch_size, len(texts))}/{len(texts)} 条')

    return results

# 6.TF-IDF + 逻辑回归基线
def tfidf_predict(texts):
    # 6.1加载向量器和模型
    pipeline = joblib.load('../../model/ml/评论情感分类.pkl')
    word_tfidf = pipeline['word_tfidf']
    char_tfidf = pipeline['char_tfidf']
    estimator = pipeline['estimator']

    # 6.2词级和字级特征拼接
    word_matrix = word_tfidf.transform([' '.join(jieba.lcut(text)) for text in texts])
    char_matrix = char_tfidf.transform(texts)
    x = hstack([word_matrix, char_matrix]).tocsr()

    # 6.3结果换回中文
    y_pred = estimator.predict(x)

    return [['负面', '中性', '正面'][i] for i in y_pred]

# 7.画图
def draw_compare(result_df, base_preds):
    labels = ['负面', '中性', '正面']

    # 7.1准确率和 macro-F1 对比
    accuracy_lora = accuracy_score(result_df['真实标签'], result_df['微调模型'])
    accuracy_tfidf = accuracy_score(result_df['真实标签'], result_df['TF-IDF模型'])
    f1_lora = f1_score(result_df['真实标签'], result_df['微调模型'], average='macro')
    f1_tfidf = f1_score(result_df['真实标签'], result_df['TF-IDF模型'], average='macro')

    names = ['LoRA微调(Qwen2.5-0.5B)', 'TF-IDF+逻辑回归']
    plt.figure(figsize=(9, 6))
    plt.bar([0 - 0.2, 1 - 0.2], [accuracy_lora, accuracy_tfidf], width=0.4, label='准确率')
    plt.bar([0 + 0.2, 1 + 0.2], [f1_lora, f1_tfidf], width=0.4, label='macro-F1')

    for i, (acc, f1) in enumerate([(accuracy_lora, f1_lora), (accuracy_tfidf, f1_tfidf)]):
        plt.text(i - 0.2, acc + 0.01, f'{acc:.4f}', ha='center')
        plt.text(i + 0.2, f1 + 0.01, f'{f1:.4f}', ha='center')

    plt.xticks([0, 1], names)
    plt.ylim(0, 1)
    plt.ylabel('得分')
    plt.title(f'大模型与机器学习模型对比（测试集 {len(result_df)} 条）')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../data/fig/finetune/微调模型对比柱状图.png')
    plt.close()

    # 7.2微调前后对比
    small = result_df.head(300)
    accuracy_base = accuracy_score(small['真实标签'], base_preds)
    accuracy_lora_small = accuracy_score(small['真实标签'], small['微调模型'])

    plt.figure(figsize=(8, 6))
    plt.bar(['微调前(基座)', '微调后(LoRA)'], [accuracy_base, accuracy_lora_small], color=['#9aa5b1', '#2f7ed8'])
    plt.text(0, accuracy_base + 0.01, f'{accuracy_base:.4f}', ha='center')
    plt.text(1, accuracy_lora_small + 0.01, f'{accuracy_lora_small:.4f}', ha='center')
    plt.ylim(0, 1)
    plt.ylabel('准确率')
    plt.title('微调前后对比（前 300 条测试数据）')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../data/fig/finetune/微调前后对比图.png')
    plt.close()

    # 7.3混淆矩阵
    cm = confusion_matrix(result_df['真实标签'], result_df['微调模型'], labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('LoRA 微调模型混淆矩阵')
    plt.tight_layout()
    plt.savefig('../../data/fig/finetune/微调模型混淆矩阵.png')
    plt.close()

# 8.读测试集
def load_test_data():
    test_df = pd.read_csv('../../data/finetune/sft_test.csv')
    test_df['真实标签'] = test_df['label'].map({-1: '负面', 0: '中性', 1: '正面'})

    return test_df, test_df['评论内容'].tolist()

# 9.基座模型测前 300 条
def run_base_model(test_df, texts, tokenizer, lora_model, label_ids):
    # 9.1关掉 LoRA
    with lora_model.disable_adapter():
        base_preds = model_score_predict(texts[:300], tokenizer, label_ids, lora_model, name='基座模型')

    print(f'\n基座模型前 300 条准确率：{accuracy_score(test_df["真实标签"][:300], base_preds):.4f}')
    # 9.2只填这 300 行
    test_df['基座模型'] = pd.Series(base_preds, index=test_df.index[:300])

    return base_preds

# 10.微调模型和基线测全部数据
def run_other_models(test_df, texts, tokenizer, lora_model, label_ids):
    test_df['微调模型'] = model_score_predict(texts, tokenizer, label_ids, lora_model, name='微调模型')
    test_df['TF-IDF模型'] = tfidf_predict(texts)

# 11.打印评估报告
def print_report(test_df):
    print(f'\n微调模型准确率：{accuracy_score(test_df["真实标签"], test_df["微调模型"]):.4f}')
    print(classification_report(test_df['真实标签'], test_df['微调模型'], labels=['负面', '中性', '正面'], digits=4))

    print(f'TF-IDF + 逻辑回归准确率：{accuracy_score(test_df["真实标签"], test_df["TF-IDF模型"]):.4f}')
    print(classification_report(test_df['真实标签'], test_df['TF-IDF模型'], labels=['负面', '中性', '正面'], digits=4))

    # 11.1一致率
    agree = (test_df['微调模型'] == test_df['TF-IDF模型']).mean()
    print(f'两个模型预测一致的比例：{agree:.4f}')


# 12.生成演示
def run_demo(tokenizer, lora_model):
    demo_texts = ['这期视频太好看了，三连支持！', '又是这种标题党，浪费时间。', '从s13一路看到s8你将得到一种升华']
    demo_preds = model_generate_predict(demo_texts, tokenizer, lora_model, name='生成演示')

    print('\n生成演示：')
    for text, pred in zip(demo_texts, demo_preds):
        print(f'{text} → {pred}')


# 13.测试
if __name__ == '__main__':
    test_df, texts = load_test_data()
    tokenizer, lora_model = get_models()
    label_ids = get_label_ids(tokenizer)

    base_preds = run_base_model(test_df, texts, tokenizer, lora_model, label_ids)
    run_other_models(test_df, texts, tokenizer, lora_model, label_ids)
    print_report(test_df)
    run_demo(tokenizer, lora_model)

    # 13.1保存明细并画图
    test_df.to_csv('../../data/finetune/sft_test_predict.csv', index=False, encoding='utf-8-sig')
    draw_compare(test_df, base_preds)
