import os
import pickle
import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import re
import warnings
warnings.filterwarnings('ignore')




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='In the future `np.object` will be defined as the corresponding NumPy scalar.')

try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError:
    pass  

# ---------------------- 核心配置---------------------
# 语料目录
CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatterbot_corpus", "data", "english")
# 语义库路径
SEMANTIC_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "english_semantic_db.pkl")
# 本地模型路径
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "all-MiniLM-L6-v2")
# 模型名称
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------- 文本归一化函数----------------------
def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    abbrev_map = {
        "what's": "what is", "who's": "who is", "where's": "where is",
        "how's": "how is", "i'm": "i am", "don't": "do not", "can't": "can not"
    }
    for abbrev, full in abbrev_map.items():
        text = text.replace(abbrev, full)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------- 批量解析所有yml语料----------------------
def parse_all_yml_files(corpus_dir):
    all_qa_pairs = []
    for filename in os.listdir(corpus_dir):
        if not filename.endswith(".yml"):
            continue
        file_path = os.path.join(corpus_dir, filename)
        print(f"🔍 正在解析：{filename}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if "conversations" in data:
                for conv in data["conversations"]:
                    if isinstance(conv, list) and len(conv) >= 2:
                        for i in range(len(conv)-1):
                            q = conv[i].strip()
                            a = conv[i+1].strip()
                            if q and a:
                                all_qa_pairs.append({"question": q, "answer": a})
        except Exception as e:
            print(f"⚠️ 解析 {filename} 失败（跳过）：{str(e)[:50]}")
    qa_df = pd.DataFrame(all_qa_pairs)
    if len(qa_df) == 0:
        raise ValueError("❌ 未解析到任何有效问答对！")
    qa_df["qa_combined"] = qa_df["question"] + "||" + qa_df["answer"]
    qa_df = qa_df.drop_duplicates(subset=["qa_combined"]).drop(columns=["qa_combined"])
    qa_df["normalized_question"] = qa_df["question"].apply(normalize_text)
    print(f"\n✅ 语料解析完成：共 {len(qa_df)} 条唯一问答对")
    return qa_df

# ---------------------- 主程序----------------------
if __name__ == "__main__":
    # 1. 批量解析所有yml文件
    qa_df = parse_all_yml_files(CORPUS_DIR)
    
    # 2. 加载本地语义模型
    print("\n🤖 加载本地语义模型...")
    # 检查本地模型文件夹是否存在
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"❌ 本地模型文件夹不存在：{LOCAL_MODEL_PATH}\n请先手动下载模型到该路径")
    # 加载本地模型
    model = SentenceTransformer(LOCAL_MODEL_PATH)
    print("✅ 本地模型加载成功！")
    
    # 3. 生成问题语义向量
    print("🔢 生成问题语义向量（可能需要1-2分钟）...")
    question_vectors = model.encode(
        qa_df["normalized_question"].tolist(),
        convert_to_tensor=False,
        show_progress_bar=True,
        batch_size=32
    )
    
    # 4. 构建KNN索引
    print("📇 构建KNN相似度索引...")
    knn_index = NearestNeighbors(
        n_neighbors=10,
        metric="cosine",
        n_jobs=-1
    ).fit(question_vectors)
    
    # 5. 保存语义库
    semantic_db = {
        "qa_df": qa_df,
        "knn_index": knn_index,
        "question_vectors": question_vectors,
        "model_name": MODEL_NAME,
        "local_model_path": LOCAL_MODEL_PATH  
    }
    with open(SEMANTIC_DB_PATH, "wb") as f:
        pickle.dump(semantic_db, f)
    
    print(f"\n🎉 全量语义库保存完成！路径：{SEMANTIC_DB_PATH}")
    print(f"📊 语义库包含 {len(qa_df)} 条问答对")