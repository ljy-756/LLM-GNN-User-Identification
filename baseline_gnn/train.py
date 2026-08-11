# baseline_gnn\train.py

import torch
import yaml
import random
import numpy as np
import os
import sys
import shutil
import json
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch_geometric.loader import DataLoader
from model.gcn import GCN
from model.gcn_strong import GCN_STRONG
from model.gat import GAT
from utils.data_loader_gnn import load_graph_data

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# 读取配置
with open("utils/configs/baseline_gnn.yaml", "r") as f:
    config = yaml.safe_load(f)

set_seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
data = load_graph_data(config["dataset_path"], feature_type=config["feature_type"])
data = data.to(device)

# 划分索引
num_nodes = data.num_nodes
num_val = int(num_nodes * config["val_ratio"])
num_test = int(num_nodes * config["test_ratio"])
num_train = num_nodes - num_val - num_test

perm = torch.randperm(num_nodes)
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.train_mask[perm[:num_train]] = True
data.val_mask[perm[num_train:num_train+num_val]] = True
data.test_mask[perm[-num_test:]] = True

# 初始化模型
if config["model_type"] == "gcn":
    model = GCN(
        input_dim=data.x.shape[1],
        hidden_dim=config["hidden_dim"],
        num_classes=2
    ).to(device)
elif config["model_type"] == "gat":
    model = GAT(
        input_dim=data.x.shape[1],
        hidden_dim=config["hidden_dim"],
        num_classes=2
    ).to(device)
elif config["model_type"] == "gcn_strong":
    model = GCN_STRONG(
        input_dim=data.x.shape[1],
        hidden_dim=config["hidden_dim"],
        num_classes=2
    ).to(device)

else:
    raise ValueError("Unknown model_type")

optimizer = torch.optim.Adam(model.parameters(),
                             lr=config["learning_rate"],
                             weight_decay=config["weight_decay"])

# 训练过程
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    logits = model(data)
    preds = logits.argmax(dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()
    return classification_report(y_true, y_pred, target_names=["human", "bot"], digits=4)

# 主训练循环
loss = 0
for epoch in range(1, config["epochs"] + 1):
    loss = train()
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")

# 评估结果
val_report = evaluate(data.val_mask)
test_report = evaluate(data.test_mask)

print("\n📊 验证集结果：")
print(val_report)
print("📊 测试集结果：")
print(test_report)

# 保存所有结果到 experiments 目录
save_dir = f"experiments/baseline_{config['model_type']}"
os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

# 保存模型
torch.save(
    model.state_dict(),
    os.path.join(save_dir, f"model/{config['model_type']}.pt")
)

# 保存分类报告
with open(os.path.join(save_dir, "classification_report.txt"), "w",encoding="utf-8") as f:
    f.write("📊 验证集结果：\n")
    f.write(val_report)
    f.write("\n📊 测试集结果：\n")
    f.write(test_report)

# 保存配置文件副本
shutil.copy("utils/configs/baseline_gnn.yaml", os.path.join(save_dir, "config.yaml"))

# 保存其他训练结果
results = {
    "final_loss": loss,
    "epochs": config["epochs"],
    "hidden_dim": config["hidden_dim"],
    "learning_rate": config["learning_rate"]
}

with open(os.path.join(save_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("✅ 模型与结果已保存至 experiments/baseline_gnn/")
