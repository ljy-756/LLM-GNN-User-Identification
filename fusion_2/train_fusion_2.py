# fusion_2\train_fusion_2.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from utils.data_loader_fusion_2 import load_graph_data_for_fusion
from model.fusion_gcn_bert_2 import FusionModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_text, data.x_graph)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data.x_text, data.x_graph)
    preds = logits.argmax(dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()
    return classification_report(y_true, y_pred, target_names=["human", "bot"], digits=4)

def main():
    with open("utils/configs/fusion_2.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    data = load_graph_data_for_fusion(
        csv_path=config["dataset_path"],
        bert_embedding_path=config["bert_embedding_path"],
        k=config["k"]
    ).to(device)
    graph_dim = data.x_graph.shape[1]

    # 划分 mask
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
    model = FusionModel(
        text_dim=config["bert_dim"],
        graph_dim=graph_dim,   
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # 训练循环
    for epoch in range(1, config["epochs"] + 1):
        loss = train(model, data, optimizer)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
    
    # 评估
    val_report = evaluate(model, data, data.val_mask)
    test_report = evaluate(model, data, data.test_mask)
    print("\n验证集结果：\n", val_report)
    print("测试集结果：\n", test_report)

    # 保存模型和分类报告
    save_dir = config["save_path"]
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "fusion_gcn_bert_2.pt"))
    with open(os.path.join(save_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("📊 验证集结果：\n")
        f.write(val_report)
        f.write("\n📊 测试集结果：\n")
        f.write(test_report)
    print(f"✅ 模型和分类报告已保存至 {save_dir}")

if __name__ == "__main__":
    main()