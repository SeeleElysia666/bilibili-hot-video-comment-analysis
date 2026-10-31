# 情感分类 LoRA 适配器

在 `Qwen/Qwen2.5-0.5B-Instruct` 上用 LoRA 微调得到的三分类适配器，用来判断 B 站评论的情感倾向（负面 / 中性 / 正面）。

## 训练配置

| 项目 | 取值 |
| --- | --- |
| 微调方式 | LoRA |
| r / alpha | 16 / 32 |
| 作用模块 | q_proj、k_proj、v_proj、o_proj |
| 训练轮数 | 2 |
| 学习率 | 2e-4 |
| 可训练参数 | 2,162,688（占总参数 0.44%） |
| 适配器大小 | 8.27 MB |

## 加载方式

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "model/finetune/lora_adapter")
```

基座模型（约 942 MB）没有放进仓库，需要自己下载，或者先放到 `model/finetune/base_model/` 下。

## 效果

在 1813 条测试样本上准确率 0.6415、macro-F1 0.6246，与基座模型、TF-IDF + 逻辑回归的完整对比见根目录 README 的 7.4 节。
