# fusion_1/extract_bert_embeddings.py

import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def extract_bert_embeddings(
    data_path,
    save_path,
    model_name="bert-base-uncased",
    batch_size=32,
    device="cpu"
):
    # ===== 0. 防止重复运行 =====
    if os.path.exists(save_path):
        print(f"⚠️ Embedding already exists at {save_path}, skip extraction.")
        return

    # ===== 1. 读取数据 =====
    df = pd.read_csv(data_path)

    # 填补空值
    df["screen_name"] = df["screen_name"].fillna("")
    df["description"] = df["description"].fillna("")

    # 拼接文本
    texts = (df["screen_name"] + " | " + df["description"]).tolist()

    # ===== 2. 加载模型 =====
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    embeddings = []

    # ===== 3. 提取 embedding =====
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT embeddings"):
            batch_texts = texts[i:i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,  # 更高效
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # ✅ 使用 CLS token（更稳定）
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()

            embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0)  # [num_samples, 768]

    # ===== 4. 归一化（关键！用于 fusion 加权）=====
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # ===== 5. 保存（带 index，防错位）=====
    save_data = {
        "embeddings": embeddings,
        "index": df.index.tolist()
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_data, save_path)

    print(f"✅ BERT embeddings saved to {save_path}")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract BERT embeddings for fusion_1")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to CSV dataset")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output path (.pt)")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    extract_bert_embeddings(
        data_path=args.data_path,
        save_path=args.save_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device
    )
