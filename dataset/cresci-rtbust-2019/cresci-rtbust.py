import pandas as pd
import json

# 读取 tweet 数据（含 user 嵌套信息）
with open("dataset/cresci-rtbust-2019/cresci-rtbust-2019_tweets.json", 'r', encoding='utf-8') as f:
    tweets_data = json.load(f)

# 提取用户信息并去重（以 user_id 为 key）
user_dict = {}
for entry in tweets_data:
    user = entry["user"]
    user_id = user["id"]
    if user_id not in user_dict:
        user_dict[user_id] = {
            "id": user_id,
            "screen_name": user.get("screen_name", ""),
            "description": user.get("description", "")
        }

# 转换为 DataFrame
user_df = pd.DataFrame(list(user_dict.values()))

# 读取标签文件（第一列是 ID，第二列是 label）
label_df = pd.read_csv("dataset/cresci-rtbust-2019/cresci-rtbust-2019.tsv", sep='\t', header=None, names=["id", "label"])

# 合并
merged = pd.merge(user_df, label_df, on="id")

# 打印样例
print("合并样例：\n", merged.head())

# 保存为 clean 版本
merged.to_csv("dataset/cresci-rtbust-2019/cresci_cleaned.csv", index=False)
merged.to_json("dataset/cresci-rtbust-2019/cresci_cleaned.json", orient="records", force_ascii=False)

print("✅ 清洗完成，已保存为 cresci_cleaned.csv/json")

