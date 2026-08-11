#utils\data_loader_gnn.py

import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
import numpy as np

def load_graph_data(path, feature_type="tfidf", k=10):
    df = pd.read_csv(path)
    df = df.dropna(subset=["description"]).reset_index(drop=True)
    texts = df["description"].astype(str).tolist()
    labels = df["label"].map({"human": 0, "bot": 1}).to_numpy()

    # TF-IDF 特征
    vectorizer = TfidfVectorizer(max_features=300)
    x = vectorizer.fit_transform(texts).toarray()
    x = torch.tensor(x, dtype=torch.float)

    # 构建 KNN 图（基于余弦相似度）
    knn_graph = kneighbors_graph(x.numpy(), k, mode='connectivity', include_self=False)
    rows, cols = knn_graph.nonzero()
    edge_index = np.vstack((rows, cols))  # shape: [2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

    # 节点标签
    y = torch.tensor(labels, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)
