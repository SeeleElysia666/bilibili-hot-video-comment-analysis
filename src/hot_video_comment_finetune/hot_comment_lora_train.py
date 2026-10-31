# 导包
import json
import torch
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 1.准备基座模型
def get_base_model():
    # 1.1读本地基座
    base_model_dir = '../../model/finetune/base_model/Qwen2.5-0.5B-Instruct'

    # 1.2加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    # 1.3设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1.4补齐方向
    tokenizer.padding_side = 'right'

    # 1.5加载模型
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.float16, device_map='auto')

    # 1.6关掉缓存
    model.config.use_cache = False

    print(f'基座模型加载完成，参数量：{model.num_parameters() / 1e9:.2f}B')

    return tokenizer, model

# 2.LoRA 配置
def get_lora_model(model):
    # 2.1秩和缩放系数
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        # 2.2挂载位置
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )

    # 2.3冻结原权重
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()   # 打印可训练参数占比

    return lora_model

# 3.数据处理
def build_dataset(tokenizer):
    # 3.1读取 jsonl
    dataset = load_dataset('json', data_files={
        'train': '../../data/finetune/sft_train.jsonl',
        'validation': '../../data/finetune/sft_valid.jsonl',
    })

    # 3.2样本转数字
    def encode_one(sample):
        messages = sample['messages']
        # 3.2.1按模板拼提示
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

        # 3.2.2答案单独编码
        answer_ids = tokenizer(messages[-1]['content'] + tokenizer.eos_token, add_special_tokens=False)['input_ids']

        # 3.2.3截断提示部分
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, truncation=True, max_length=256 - len(answer_ids)
        )['input_ids']

        # 3.2.4labels 填 -100
        labels = [-100] * len(prompt_ids) + answer_ids
        input_ids = prompt_ids + answer_ids

        return {
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'labels': labels,
        }

    dataset = dataset.map(encode_one, remove_columns=dataset['train'].column_names)
    print(dataset)

    return dataset

# 4.拼 batch
def collate_fn(batch, pad_id=0):
    max_len = max(len(item['input_ids']) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        input_ids.append(item['input_ids'] + [pad_id] * pad_len)
        attention_mask.append(item['attention_mask'] + [0] * pad_len)
        labels.append(item['labels'] + [-100] * pad_len)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }

# 5.训练
def train_lora(lora_model, tokenizer, dataset):
    # 5.1训练参数
    training_args = TrainingArguments(
        output_dir='../../output/lora_train',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        # 热身步数
        warmup_steps=80,
        logging_steps=20,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        fp16=True,
        report_to='none',
        remove_unused_columns=False,
    )

    # 5.2Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        # 5.3补齐参数
        data_collator=partial(collate_fn, pad_id=tokenizer.pad_token_id),
    )

    # 5.4开始训练
    trainer.train()

    # 5.5保存适配器
    lora_model.save_pretrained('../../model/finetune/lora_adapter')
    tokenizer.save_pretrained('../../model/finetune/lora_adapter')
    print('LoRA 适配器已保存：../../model/finetune/lora_adapter')

    # 5.6保存日志
    with open('../../output/lora_train_log.json', 'w', encoding='utf-8') as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print('训练日志已保存：../../output/lora_train_log.json')

# 6.测试
if __name__ == '__main__':
    tokenizer, model = get_base_model()
    lora_model = get_lora_model(model)
    dataset = build_dataset(tokenizer)
    train_lora(lora_model, tokenizer, dataset)
