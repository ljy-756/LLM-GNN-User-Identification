# fusion_two_labels/extract_bert_embeddings.py

import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def extract_bert_embeddings(data_path, text_column, save_path, model_name="bert-base-uncased", batch_size=32, device="cpu"):
    # 读取数据
    df = pd.read_csv(data_path)

    # 填补空值
    df["screen_name"] = df["screen_name"].fillna("")
    df["description"] = df["description"].fillna("")

    # 拼接文本作为输入
    texts = (df["screen_name"] + " | " + df["description"]).tolist()

    # 加载预训练BERT模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT embeddings"):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=192, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # 取 pooler_output 作为句子embedding，shape [batch_size, 768]
            batch_embeddings = outputs.pooler_output.cpu()
            embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0)  # [num_samples, 768]

    torch.save(embeddings, save_path)
    print(f"BERT embeddings saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract BERT embeddings for fusion GCN")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--text_column", type=str, default="combined_text", help="Text column name")
    parser.add_argument("--save_path", type=str, required=True, help="Output path for embeddings (.pt)")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained BERT model name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    extract_bert_embeddings(args.data_path, args.text_column, args.save_path, args.model_name, args.batch_size, args.device)
