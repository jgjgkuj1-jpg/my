import os
import pickle
import warnings
import re
import sys

import tkinter as tk
from tkinter import ttk, scrolledtext, font, messagebox

import ctypes
from sentence_transformers import SentenceTransformer

# 忽略无关警告
warnings.filterwarnings('ignore')

# ---------------------- 系统适配（让界面更清晰）----------------------
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 适配高分屏
except:
    pass

# ---------------------- 核心配置 ----------------------
def get_base_dir():
    if hasattr(sys, '_MEIPASS'):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return base_dir

BASE_DIR = get_base_dir()
SEMANTIC_DB_PATH = os.path.join(BASE_DIR, "english_semantic_db.pkl")
# 本地模型路径（和语义库构建脚本一致）
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = 0.6
LOW_SIM_THRESHOLD = 0.3

# ---------------------- 加载模型 ----------------------
print("✅ 正在加载已训练的语义库和模型...")
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

print("🎉 模型加载完成！高颜值聊天界面即将启动...")
print(f"📌 本地模型路径：{LOCAL_MODEL_PATH}")

# ---------------------- 文本归一化函数 ----------------------
def normalize_text(text):
    """文本归一化：小写、去标点、展开缩写、去多余空格"""
    if not text:
        return ""
    # 转小写
    text = text.lower()
    # 去除英文标点（保留空格）
    text = re.sub(r'[^\w\s]', '', text)
    # 展开常见缩写（适配botprofile数据集）
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
    # 去多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------- 核心回答逻辑 ----------------------
def get_bot_response(user_input):
    import random
    if not user_input.strip():
        return "Please enter a valid English question!"
    
    try:
        # 归一化用户输入
        normalized_input = normalize_text(user_input.strip())
        # 生成输入的语义向量
        input_vector = model.encode(normalized_input, convert_to_tensor=False).reshape(1, -1)
        
        # 查找Top10相似问题
        distances, indices = semantic_db["knn_index"].kneighbors(input_vector)
        similarities = 1 - distances[0]  # 余弦距离转相似度（0-1）
        
        # 1. 高相似度（≥0.6）：从相似答案中随机选
        high_sim_indices = [idx for idx, sim in zip(indices[0], similarities) if sim >= 0.6]
        if high_sim_indices:
            high_sim_answers = [semantic_db["qa_df"].iloc[idx]["answer"] for idx in high_sim_indices]
            return random.choice(high_sim_answers)
        
        # 2. 中相似度（0.3~0.6）：从Top5相似答案中随机选
        mid_sim_indices = [idx for idx, sim in zip(indices[0], similarities) if 0.3 <= sim < 0.6]
        if mid_sim_indices:
            mid_sim_answers = [semantic_db["qa_df"].iloc[idx]["answer"] for idx in mid_sim_indices[:5]]
            random_ans = random.choice(mid_sim_answers)
            return f"Based on related topics: {random_ans} Maybe this helps!"
        
        # 3. 低相似度/无匹配：从全量语料中随机选一个兜底
        all_answers = semantic_db["qa_df"]["answer"].tolist()
        random_default = random.choice(all_answers)
        return f"Sorry, I don't have a direct answer for that. Here's something fun: {random_default}"
    
    except Exception as e:
        # 异常时也随机返回一个答案
        all_answers = semantic_db["qa_df"]["answer"].tolist()
        random_error = random.choice(all_answers)
        return f"Oops! Something went wrong: {str(e)[:50]}\nRandom reply: {random_error}"

# ---------------------- 界面类 ----------------------
class BeautifulChatBot:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 English Smart ChatBot")
        self.root.geometry("1920x1080")
        self.root.minsize(700, 500)  # 最小窗口尺寸
        self.root.resizable(True, True)
        
        # ---------------------- 自定义样式 ----------------------
        # 1. 自定义字体
        self.font_title = font.Font(family="Microsoft YaHei", size=16, weight="bold")
        self.font_normal = font.Font(family="Microsoft YaHei", size=12)
        self.font_small = font.Font(family="Microsoft YaHei", size=10)
        
        # 2. 颜色配置（莫兰迪色系）
        self.COLOR_BG = "#f0f5ff"  # 背景色
        self.COLOR_USER = "#4096ff"  # 用户消息色
        self.COLOR_BOT = "#67c23a"   # 机器人消息色
        self.COLOR_BTN = "#1890ff"   # 按钮色
        self.COLOR_BTN_HOVER = "#40a9ff"  # 按钮悬停色
        self.COLOR_FRAME = "#ffffff" # 框架背景
        self.COLOR_TEXT = "#303133"  # 正文色
        
        # 3. 设置主背景
        self.root.configure(bg=self.COLOR_BG)
        
        # ---------------------- 顶部标题栏 ----------------------
        self.title_frame = tk.Frame(root, bg=self.COLOR_BG)
        self.title_frame.pack(fill=tk.X, padx=20, pady=15)
        
        self.title_label = tk.Label(
            self.title_frame,
            text="🤖 English Smart ChatBot",
            font=self.font_title,
            bg=self.COLOR_BG,
            fg="#1f2937"
        )
        self.title_label.pack(side=tk.LEFT)
        
        self.subtitle_label = tk.Label(
            self.title_frame,
            text="基于你的英文数据集训练 · 支持不同问法识别",
            font=self.font_small,
            bg=self.COLOR_BG,
            fg="#666666"
        )
        self.subtitle_label.pack(side=tk.LEFT, padx=15)
        
        # ---------------------- 聊天记录框----------------------
        self.chat_frame = tk.Frame(root, bg=self.COLOR_BG)
        self.chat_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # 圆角聊天框（用Label模拟边框）
        self.chat_border = tk.Label(
            self.chat_frame,
            bg="#e8f4f8",
            relief=tk.RAISED,
            bd=1
        )
        self.chat_border.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.chat_history = scrolledtext.ScrolledText(
            self.chat_border,
            wrap=tk.WORD,
            font=self.font_normal,
            bg=self.COLOR_FRAME,
            fg=self.COLOR_TEXT,
            padx=15,
            pady=15,
            bd=0,
            highlightthickness=0
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.chat_history.config(state=tk.DISABLED)
        
        # 设置消息标签样式
        self.chat_history.tag_config("user", foreground=self.COLOR_USER, font=(self.font_normal, 12, "bold"))
        self.chat_history.tag_config("user_text", foreground="#333333", font=self.font_normal)
        self.chat_history.tag_config("bot", foreground=self.COLOR_BOT, font=(self.font_normal, 12, "bold"))
        self.chat_history.tag_config("bot_text", foreground="#333333", font=self.font_normal)
        self.chat_history.tag_config("time", foreground="#999999", font=self.font_small)
        
        # ---------------------- 输入区域----------------------
        self.input_frame = tk.Frame(root, bg=self.COLOR_BG)
        self.input_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # 输入框（圆角+阴影）
        self.input_border = tk.Label(self.input_frame, bg="#dcdfe6")
        self.input_border.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        
        self.user_input = tk.Entry(
            self.input_border,
            font=self.font_normal,
            bg=self.COLOR_FRAME,
            fg=self.COLOR_TEXT,
            bd=0,
            highlightthickness=0,
            insertbackground=self.COLOR_BTN,
            relief=tk.FLAT
        )
        self.user_input.pack(fill=tk.X, expand=True, padx=10, pady=8)
        self.user_input.bind("<Return>", self.send_message)
        self.user_input.bind("<FocusIn>", lambda e: self.user_input.config(bg="#f8f9fa"))
        self.user_input.bind("<FocusOut>", lambda e: self.user_input.config(bg=self.COLOR_FRAME))
        self.user_input.insert(0, "请输入英文问题（如：What is AI?）...")
        self.user_input.bind("<Button-1>", lambda e: self.clear_placeholder())
        
        # 按钮容器
        self.btn_frame = tk.Frame(self.input_frame, bg=self.COLOR_BG)
        self.btn_frame.pack(side=tk.RIGHT, padx=10)
        
        # 发送按钮
        self.send_btn = tk.Button(
            self.btn_frame,
            text="发送",
            font=self.font_normal,
            bg=self.COLOR_BTN,
            fg="white",
            bd=0,
            padx=20,
            pady=8,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.send_btn.pack(side=tk.LEFT, padx=5)
        self.send_btn.bind("<Enter>", lambda e: self.send_btn.config(bg=self.COLOR_BTN_HOVER))
        self.send_btn.bind("<Leave>", lambda e: self.send_btn.config(bg=self.COLOR_BTN))
        self.send_btn.config(command=self.send_message)
        
        # 清空按钮
        self.clear_btn = tk.Button(
            self.btn_frame,
            text="清空",
            font=self.font_small,
            bg="#f5f7fa",
            fg="#666666",
            bd=0,
            padx=15,
            pady=8,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.clear_btn.pack(side=tk.LEFT)
        self.clear_btn.bind("<Enter>", lambda e: self.clear_btn.config(bg="#e5e6eb"))
        self.clear_btn.bind("<Leave>", lambda e: self.clear_btn.config(bg="#f5f7fa"))
        self.clear_btn.config(command=self.clear_history)
        
        # 初始消息
        self.add_message("Bot", "Hello! I'm your English chatbot. Ask me anything!")

    # ---------------------- 辅助函数 ----------------------


    def clear_placeholder(self):
        """清空输入框占位符"""
        if self.user_input.get() == "请输入英文问题（如：What is AI?）...":
            self.user_input.delete(0, tk.END)
            self.user_input.config(fg=self.COLOR_TEXT)

    def add_message(self, sender, message):
        """添加美化后的消息"""
        self.chat_history.config(state=tk.NORMAL)
        
        # 格式化消息
        if sender == "You":
            self.chat_history.insert(tk.END, "👤 你: ", "user")
            self.chat_history.insert(tk.END, f"{message}\n\n", "user_text")
        else:
            self.chat_history.insert(tk.END, "🤖 机器人: ", "bot")
            self.chat_history.insert(tk.END, f"{message}\n\n", "bot_text")
        
        # 滚动到底部
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def send_message(self, event=None):
        """发送消息（新增英文输入检测）"""
        user_text = self.user_input.get().strip()
        
        # 1. 空值/默认提示文本检测
        if not user_text or user_text == "请输入英文问题（如：What is AI?）...":
            messagebox.showwarning("提示", "请输入有效的英文问题！")
            return
        
        # 2. 英文输入检测
        # 正则匹配任意英文字母（a-z/A-Z）
        has_english = re.search(r'[a-zA-Z]', user_text)
        if not has_english:
            messagebox.showwarning("提示", "请输入以英文为主的问题（可包含数字/英文标点）！")
            return
        
        # 3. 正常流程
        self.add_message("You", user_text)
        self.user_input.delete(0, tk.END)
        
        bot_response = get_bot_response(user_text)
        self.add_message("Bot", bot_response)

    def clear_history(self):
        """清空聊天记录"""
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state=tk.DISABLED)
        self.add_message("Bot", "Hello! I'm your English chatbot. Ask me anything!")

# ---------------------- 启动界面 ----------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = BeautifulChatBot(root)
    root.mainloop()