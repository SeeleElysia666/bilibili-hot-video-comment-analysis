# 导包
import jieba
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1.数据准备
def build_vocab(texts):
    # 1.1统计词频
    counter = {}
    for text in texts:
        for word in jieba.lcut(text):
            counter[word] = counter.get(word, 0) + 1

    words = sorted(counter.items(), key=lambda x: -x[1])[:9998]
    # 1.2词表编号
    word_dict = {word: index + 2 for index, (word, _) in enumerate(words)}

    return word_dict

def encode_texts(texts, word_dict):
    # 1.3评论转编号
    all_ids = []
    for text in texts:
        ids = [word_dict.get(word, 1) for word in jieba.lcut(text)][:64]
        ids = ids + [0] * (64 - len(ids))
        all_ids.append(ids)

    return torch.tensor(all_ids, dtype=torch.long)

# 2.初始化模型参数
def init_params(device):
    # 2.1固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 2.2词向量矩阵
    params = {'embedding': torch.randn(10000, 64, device=device) * 0.1}

    # 2.3三种卷积核的权重和偏置
    for size in [2, 3, 4]:
        params[f'conv{size}_w'] = torch.randn(100, 64, size, device=device) * 0.1
        params[f'conv{size}_b'] = torch.zeros(100, device=device)

    # 2.4全连接层
    params['fc_w'] = torch.randn(3, 300, device=device) * 0.1
    params['fc_b'] = torch.zeros(3, device=device)

    # 2.5转成可训练参数
    return {name: torch.nn.Parameter(value) for name, value in params.items()}

# 3.前向传播
def forward(params, x, training=False):
    # 3.1查词向量，维度换成卷积要的（批, 64, 句长）
    emb = F.embedding(x, params['embedding'], padding_idx=0).transpose(1, 2)

    # 3.2三路卷积，各自取最大值
    outs = []
    for size in [2, 3, 4]:
        conv = F.conv1d(emb, params[f'conv{size}_w'], params[f'conv{size}_b'], padding=size // 2)
        outs.append(F.relu(conv).max(dim=2).values)

    # 3.3拼接后过 dropout 和全连接
    out = torch.cat(outs, dim=1)
    out = F.dropout(out, 0.3, training=training)

    return F.linear(out, params['fc_w'], params['fc_b'])

# 4.训练
def train_model(params, train_x, train_y, device):
    # 4.1切验证集
    split = int(len(train_x) * 0.9)
    valid_x, valid_y = train_x[split:], train_y[split:]
    train_x, train_y = train_x[:split], train_y[:split]

    # 4.2损失函数和优化器
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params.values(), lr=1e-3)

    history = []
    # 4.3训练轮数
    for epoch in range(5):
        # 4.4打乱顺序
        order = torch.randperm(len(train_x))
        total_loss = 0

        for start in range(0, len(train_x), 64):
            index = order[start:start + 64]
            batch_x, batch_y = train_x[index].to(device), train_y[index].to(device)

            # 4.5前向、反向、更新参数
            optimizer.zero_grad()
            output = forward(params, batch_x, training=True)
            loss = loss_func(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_x)

        # 4.6验证集评估
        with torch.no_grad():
            valid_pred = forward(params, valid_x.to(device)).argmax(dim=1).cpu().numpy()

        avg_loss = total_loss / len(train_x)
        valid_acc = accuracy_score(valid_y.numpy(), valid_pred)
        history.append({'epoch': epoch + 1, 'loss': avg_loss, 'valid_acc': valid_acc})
        print(f'第 {epoch + 1} 轮：训练损失 {avg_loss:.4f}，验证集准确率 {valid_acc:.4f}')

    return history

# 5.画训练曲线
def draw_history(history):
    epochs_list = [item['epoch'] for item in history]
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_list, [item['loss'] for item in history], marker='o', label='训练损失')
    plt.plot(epochs_list, [item['valid_acc'] for item in history], marker='s', label='验证集准确率')
    plt.xlabel('训练轮数')
    plt.title('TextCNN 训练过程')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../data/fig/dl/TextCNN训练曲线.png')
    plt.close()

# 6.读数据
def load_train_test_data():
    train_df = pd.concat([pd.read_json('../../data/finetune/sft_train.jsonl', lines=True),
                          pd.read_json('../../data/finetune/sft_valid.jsonl', lines=True)], ignore_index=True)
    test_df = pd.read_csv('../../data/finetune/sft_test.csv')

    # 6.1取出评论和标签
    label_dict = {-1: '负面', 0: '中性', 1: '正面'}
    labels = ['负面', '中性', '正面']
    train_texts = [item[1]['content'].split('：', 1)[1] for item in train_df['messages']]
    train_labels = [labels.index(item[2]['content']) for item in train_df['messages']]
    test_texts = test_df['评论内容'].tolist()
    test_labels = [labels.index(label_dict[value]) for value in test_df['label']]

    # 6.2建词表并转张量
    word_dict = build_vocab(train_texts)
    print(f'词表大小：{len(word_dict)}')

    train_x = encode_texts(train_texts, word_dict)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    test_x = encode_texts(test_texts, word_dict)

    return train_x, train_y, test_x, test_labels

# 7.测试集评估
def evaluate_model(params, test_x, test_labels, device):
    with torch.no_grad():
        pred = forward(params, test_x.to(device)).argmax(dim=1).cpu().numpy()

    print(f'\n测试集准确率：{accuracy_score(test_labels, pred):.4f}')
    print(f'测试集 macro-F1：{f1_score(test_labels, pred, average="macro"):.4f}')
    print(classification_report(test_labels, pred, target_names=['负面', '中性', '正面'], digits=4))

# 8.保存模型和曲线
def save_model(params, history):
    torch.save({name: value.detach().cpu() for name, value in params.items()}, '../../model/dl/textcnn.pth')
    draw_history(history)

# 9.测试
if __name__ == '__main__':
    train_x, train_y, test_x, test_labels = load_train_test_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = init_params(device)
    history = train_model(params, train_x, train_y, device)
    evaluate_model(params, test_x, test_labels, device)
    save_model(params, history)
