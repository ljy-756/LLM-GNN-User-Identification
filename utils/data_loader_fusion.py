# utils/data_loader_fusion.py

import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import numpy as np

def load_graph_data_with_bert_embedding(csv_path, bert_embedding_path, k=10):
    """
    加载数据，使用预先提取的BERT embedding作为节点特征，
    构建基于embedding的KNN图。

    参数：
        csv_path: 含有label和文本的csv路径（必须含label列）
        bert_embedding_path: 预先提取好的BERT embedding的.pt文件路径
        k: KNN图邻居数量

    返回：
        PyG Data对象，包含 x, edge_index, y
    """
    # 读取CSV标签
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    # 标签映射 human->0, bot->1
    labels = df["label"].map({"human": 0, "bot": 1}).to_numpy()
    y = torch.tensor(labels, dtype=torch.long)

    # 加载预提取的BERT embedding，形状应为 [num_nodes, embedding_dim]
    x = torch.load(bert_embedding_path)
    assert x.shape[0] == len(df), "Embedding数量与数据行数不匹配"

    # 构建KNN图（基于embedding的余弦相似度）
    knn_graph = kneighbors_graph(x.numpy(), n_neighbors=k, mode='connectivity', include_self=False)
    rows, cols = knn_graph.nonzero()
    edge_index = np.vstack((rows, cols))
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

    return Data(x=x, edge_index=edge_index, y=y)
