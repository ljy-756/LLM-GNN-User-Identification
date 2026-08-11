# baseline_text/train_bert_classifier.py

import pandas as pd
import torch
import random
import os
import sys
import yaml
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    BertTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer
)
from datasets import Dataset
from model.bert_classifier import BertClassifier
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =========================
# 读取配置
# =========================
with open("utils/configs/baseline_text.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# =========================
# 固定随机种子
# =========================
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# =========================
# 加载数据
# =========================
df = pd.read_csv(config["data_path"])

df["screen_name"] = df["screen_name"].fillna("")
df["description"] = df["description"].fillna("")
df["combined_text"] = df["screen_name"] + " | " + df["description"]

# 保留需要的列（使用配置中的标签列）
df = df[["combined_text", config["label_column"]]].dropna()

# 标签映射
df[config["label_column"]] = df[config["label_column"]].map({"human": 0, "bot": 1})

# =========================
# 数据划分：train / val / test
# =========================
train_df, temp_df = train_test_split(
    df,
    test_size=config["validation_split"] + config["test_split"],
    random_state=config["seed"],
    stratify=df[config["label_column"]]
)

val_ratio_adjusted = config["validation_split"] / (config["validation_split"] + config["test_split"])
val_df, test_df = train_test_split(
    temp_df,
    test_size=1 - val_ratio_adjusted,
    random_state=config["seed"],
    stratify=temp_df[config["label_column"]]
)

# 转换为 HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# =========================
# 类别权重（手动设置，可调整）
# =========================
# 原本使用 compute_class_weight 自动计算，但此处手动覆盖为 [1.0, 4.0] (human=0, bot=1)
class_weights = torch.tensor([1.0, 4.0], dtype=torch.float)
device = "cuda" if torch.cuda.is_available() else "cpu"
class_weights = class_weights.to(device)
print("类别权重（手动设置 human=1.0, bot=4.0）：", class_weights.tolist())

# =========================
# Tokenizer 和 tokenize 函数
# =========================
tokenizer = BertTokenizer.from_pretrained(config["model_name"])

# 文本字段：使用配置中的 text_column，若不存在则用 "combined_text"
text_col = config.get("text_column", "combined_text")

def tokenize_function(example):
    return tokenizer(
        example[text_col],
        truncation=True,
        max_length=config["max_length"]
    )

# 对三个数据集分别进行 tokenize
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 设置格式为 torch 张量
columns = ['input_ids', 'attention_mask', config["label_column"]]
train_dataset.set_format(type='torch', columns=columns)
val_dataset.set_format(type='torch', columns=columns)
test_dataset.set_format(type='torch', columns=columns)

# =========================
# 模型初始化
# =========================
model = BertClassifier(model_name=config["model_name"], num_labels=2)
model.config.problem_type = "single_label_classification"

# =========================
# 自定义 Trainer（支持类别权重）
# =========================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# =========================
# 评估指标函数
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "eval_accuracy": accuracy_score(labels, preds),
        "eval_precision": precision_score(labels, preds, average='binary', pos_label=1, zero_division=0),
        "eval_recall": recall_score(labels, preds, average='binary', pos_label=1, zero_division=0),
        "eval_f1": f1_score(labels, preds, average='binary', pos_label=1, zero_division=0),
    }

# =========================
# 训练参数
# =========================
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    num_train_epochs=config["epochs"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    logging_dir=os.path.join(config["output_dir"], "logs"),
)

# =========================
# 创建 Trainer
# =========================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
)

# =========================
# 开始训练
# =========================
trainer.train()

# =========================
# 保存模型和 tokenizer
# =========================
model.save_pretrained(config["save_path"])
tokenizer.save_pretrained(config["save_path"])
print("✅ 模型已保存至：", config["save_path"])

# =========================
# 最终评估：验证集
# =========================
val_preds = trainer.predict(val_dataset)
val_pred_labels = val_preds.predictions.argmax(-1)
val_true_labels = val_dataset[config["label_column"]].numpy()

val_report = classification_report(val_true_labels, val_pred_labels, target_names=["human", "bot"], zero_division=0)
print("\n📊 验证集性能：")
print(val_report)

# =========================
# 最终评估：测试集
# =========================
test_preds = trainer.predict(test_dataset)
test_pred_labels = test_preds.predictions.argmax(-1)
test_true_labels = test_dataset[config["label_column"]].numpy()

test_report = classification_report(test_true_labels, test_pred_labels, target_names=["human", "bot"], zero_division=0)
print("\n📊 测试集性能：")
print(test_report)

# =========================
# 保存评估结果到文件
# =========================
save_dir = "experiments/baseline_text_bert"
os.makedirs(save_dir, exist_ok=True)

# 保存验证集报告
with open(os.path.join(save_dir, "classification_report_val.txt"), "w", encoding="utf-8") as f:
    f.write("验证集分类报告：\n")
    f.write(val_report)

# 保存测试集报告
with open(os.path.join(save_dir, "classification_report_test.txt"), "w", encoding="utf-8") as f:
    f.write("测试集分类报告：\n")
    f.write(test_report)

# 保存详细的指标（验证集和测试集）
metrics = {
    "val": {
        "accuracy": accuracy_score(val_true_labels, val_pred_labels),
        "precision": precision_score(val_true_labels, val_pred_labels, pos_label=1, zero_division=0),
        "recall": recall_score(val_true_labels, val_pred_labels, pos_label=1, zero_division=0),
        "f1": f1_score(val_true_labels, val_pred_labels, pos_label=1, zero_division=0),
    },
    "test": {
        "accuracy": accuracy_score(test_true_labels, test_pred_labels),
        "precision": precision_score(test_true_labels, test_pred_labels, pos_label=1, zero_division=0),
        "recall": recall_score(test_true_labels, test_pred_labels, pos_label=1, zero_division=0),
        "f1": f1_score(test_true_labels, test_pred_labels, pos_label=1, zero_division=0),
    }
}

with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("✅ 评估报告已保存至：", save_dir)