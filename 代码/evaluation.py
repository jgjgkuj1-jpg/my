# 导入所需库
import numpy as np
import re
import sys
import os
import pickle
import warnings
import random
import ctypes
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

# ---------------------- 1. 整合机器人核心代码 ----------------------
# 下载nltk所需资源
nltk.download('punkt_tab')

# 忽略无关警告
warnings.filterwarnings('ignore')

# ---------------------- 系统适配 ----------------------
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# ---------------------- 核心配置----------------------
def get_base_dir():
    if hasattr(sys, '_MEIPASS'):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return base_dir

BASE_DIR = get_base_dir()
SEMANTIC_DB_PATH = os.path.join(BASE_DIR, "english_semantic_db.pkl")
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = 0.6
LOW_SIM_THRESHOLD = 0.3

# ---------------------- 加载模型和语义库 ----------------------
print("✅ 正在加载机器人依赖的语义库和模型...")
if not os.path.exists(SEMANTIC_DB_PATH):
    raise FileNotFoundError(
        f"语义库文件不存在！\n"
        f"当前查找路径：{SEMANTIC_DB_PATH}\n"
        f"请先运行语义库构建脚本生成该文件"
    )

# 加载语义库
with open(SEMANTIC_DB_PATH, "rb") as f:
    semantic_db = pickle.load(f)

# 加载本地模型
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"❌ 本地模型文件夹不存在：{LOCAL_MODEL_PATH}")
model = SentenceTransformer(LOCAL_MODEL_PATH)

# ---------------------- 文本归一化函数 ----------------------
def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    abbrev_map = {
        "what's": "what is",
        "who's": "who is",
        "where's": "where is",
        "how's": "how is",
        "i'm": "i am",
        "don't": "do not",
        "can't": "can not"
    }
    for abbrev, full in abbrev_map.items():
        text = text.replace(abbrev, full)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------- 核心回答逻辑 ----------------------
def get_bot_response(user_input):
    if not user_input.strip():
        return "Please enter a valid English question!"
    
    try:
        normalized_input = normalize_text(user_input.strip())
        input_vector = model.encode(normalized_input, convert_to_tensor=False).reshape(1, -1)
        
        distances, indices = semantic_db["knn_index"].kneighbors(input_vector)
        similarities = 1 - distances[0]
        
        # 高相似度匹配
        high_sim_indices = [idx for idx, sim in zip(indices[0], similarities) if sim >= 0.6]
        if high_sim_indices:
            high_sim_answers = [semantic_db["qa_df"].iloc[idx]["answer"] for idx in high_sim_indices]
            return random.choice(high_sim_answers)
        
        # 中相似度匹配
        mid_sim_indices = [idx for idx, sim in zip(indices[0], similarities) if 0.3 <= sim < 0.6]
        if mid_sim_indices:
            mid_sim_answers = [semantic_db["qa_df"].iloc[idx]["answer"] for idx in mid_sim_indices[:5]]
            random_ans = random.choice(mid_sim_answers)
            return f"Based on related topics: {random_ans} Maybe this helps!"
        
        # 低相似度兜底
        all_answers = semantic_db["qa_df"]["answer"].tolist()
        random_default = random.choice(all_answers)
        return f"Sorry, I don't have a direct answer for that. Here's something fun: {random_default}"
    
    except Exception as e:
        all_answers = semantic_db["qa_df"]["answer"].tolist()
        random_error = random.choice(all_answers)
        return f"Oops! Something went wrong: {str(e)[:50]}\nRandom reply: {random_error}"

# ---------------------- 2. 评估相关函数 ----------------------
def calculate_semantic_similarity(question, bot_response, reference_answer):
    """计算语义相似度（复用机器人的模型）"""
    text1 = f"{question} {bot_response}"
    text2 = f"{question} {reference_answer}"
    emb1 = model.encode(text1, convert_to_tensor=True) 
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    return round(similarity, 4)

def calculate_bleu_score(bot_response, reference_answer):
    """计算BLEU分数"""
    if not isinstance(bot_response, str) or not isinstance(reference_answer, str):
        return 0.0
    # 清理机器人回复中的额外提示文本（只保留核心答案）
    bot_resp_clean = re.sub(r'Based on related topics: |Maybe this helps!|Sorry, I don\'t have a direct answer for that. Here\'s something fun: |Oops! Something went wrong: .*\nRandom reply: ', '', bot_response)
    # 分词计算
    reference_tokens = word_tokenize(reference_answer.strip().lower())
    candidate_tokens = word_tokenize(bot_resp_clean.strip().lower())
    if len(reference_tokens) == 0 or len(candidate_tokens) == 0:
        return 0.0
    smooth_func = SmoothingFunction().method4
    bleu = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smooth_func)
    return round(bleu, 4)

# ---------------------- 3. 测试集配置----------------------
# 格式：[("用户英文问题", "人工标准答案"), ...]

TEST_DATA = [
    ("What is artificial intelligence?", "AI is a branch of computer science that simulates human intelligence in machines."),
    ("How to learn Python?", "You can learn Python through online courses, practice projects and reading official documents."),
    ("What is the purpose of chatbots?", "Chatbots are designed to interact with humans and answer their questions automatically."),
    ("Can machines think?", "Machines can simulate thinking processes but do not have real consciousness like humans."),
    ("What is machine learning?", "Machine learning is a subset of AI that enables systems to learn from data without explicit programming.")
]

# ---------------------- 4. 生成机器人真实回复 ----------------------
# 遍历测试集问题，调用机器人真实回复函数生成回复
BOT_RESPONSES = []
print("🔄 正在生成机器人回复...")
for question, _ in TEST_DATA:
    bot_resp = get_bot_response(question)
    BOT_RESPONSES.append(bot_resp)
print("✅ 机器人回复生成完成！")

# ---------------------- 5. 批量评估计算 ----------------------
similarity_scores = []
bleu_scores = []
qualified_samples = 0
EVAL_SIM_THRESHOLD = 0.5  # 评估用的合格阈值

print("\n===== 单样本评估结果 =====")
print(f"{'问题':<50} | {'语义相似度':<10} | {'BLEU分数':<10} | {'是否合格'}")
print("-" * 100)

for idx, ((question, reference), bot_resp) in enumerate(zip(TEST_DATA, BOT_RESPONSES)):
    try:
        sim_score = calculate_semantic_similarity(question, bot_resp, reference)
        bleu_score = calculate_bleu_score(bot_resp, reference)
    except Exception as e:
        sim_score = 0.0
        bleu_score = 0.0
        print(f"⚠️  第{idx+1}条样本计算异常：{str(e)}", file=sys.stderr)
    
    similarity_scores.append(sim_score)
    bleu_scores.append(bleu_score)
    
    is_qualified = "是" if sim_score >= EVAL_SIM_THRESHOLD else "否"
    if is_qualified == "是":
        qualified_samples += 1
    
    # 格式化输出（截断长问题）
    short_question = question[:47] + "..." if len(question) > 50 else question
    print(f"{short_question:<50} | {sim_score:<10} | {bleu_score:<10} | {is_qualified}")

# ---------------------- 6. 汇总结果 ----------------------
if len(similarity_scores) > 0:
    avg_similarity = round(np.mean(similarity_scores), 4)
    std_similarity = round(np.std(similarity_scores), 4)
    avg_bleu = round(np.mean(bleu_scores), 4)
    std_bleu = round(np.std(bleu_scores), 4)
    accuracy = round((qualified_samples / len(TEST_DATA)) * 100, 2)
else:
    avg_similarity = avg_bleu = std_similarity = std_bleu = accuracy = 0.0

print("\n===== 实验汇总结果 =====")
print(f"测试集总样本数：{len(TEST_DATA)}")
print(f"平均语义相似度得分：{avg_similarity}（标准差：{std_similarity}）")
print(f"平均BLEU分数：{avg_bleu}（标准差：{std_bleu}）")
print(f"准确率（语义相似度≥{EVAL_SIM_THRESHOLD}）：{accuracy}%（{qualified_samples}/{len(TEST_DATA)}）")