# utils/data_loader_fusion_3.py

import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def load_graph_data_for_fusion(
    csv_path,
    bert_embedding_path,
    k=5
):
    # ===== 1. 读取CSV =====
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    df["screen_name"] = df["screen_name"].fillna("")
    df["description"] = df["description"].fillna("")

    texts = (df["screen_name"] + " " + df["description"]).tolist()

    # ===== 2. 标签 =====
    labels = df["label"].map({"human": 0, "bot": 1}).to_numpy()
    y = torch.tensor(labels, dtype=torch.long)

    # ===== 3. BERT embedding =====
    bert_data = torch.load(bert_embedding_path)

    # 如果加载出来的是 dict，就取 "embeddings"；否则直接用它
    if isinstance(bert_data, dict) and "embeddings" in bert_data:
        bert_x = bert_data["embeddings"]
    else:
        bert_x = bert_data

    # 检查数量是否和 CSV 行数一致
    assert bert_x.shape[0] == len(df), f"Embedding数量({bert_x.shape[0]})与数据行数({len(df)})不匹配"

    # ===== 4. graph feature（TF-IDF）=====
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    graph_x = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float)

    # 归一化（重要）
    graph_x = torch.nn.functional.normalize(graph_x, p=2, dim=1)

    # ===== 5. KNN图（基于 graph feature）=====
    knn_graph = kneighbors_graph(
        graph_x.numpy(),
        n_neighbors=k,
        mode='connectivity',
        include_self=False
    )

    rows, cols = knn_graph.nonzero()
    edge_index = torch.tensor(
        np.vstack((rows, cols)),
        dtype=torch.long
    ).contiguous()

    # ===== 6. 返回多特征 =====
    data = Data(
        x=bert_x,
        edge_index=edge_index,
        y=y
    )
    data.x_text = bert_x          # [N, 768]
    data.x_graph = graph_x
    
    return data