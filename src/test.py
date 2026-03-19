import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

#  设置中文字体 
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ========== 1. 定义数据集类 ==========
class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['covid19', 'lungOpacity', 'normal', 'pneumonia']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                raise FileNotFoundError(f"类别文件夹不存在: {cls_dir}")
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path  # 返回图像路径，用于可视化

# ========== 2. 加载测试集数据 ==========
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.226])
])

test_dataset = ChestXRayDataset(
    root_dir='./dataset/test',
    transform=test_transform
)

batch_size = 4
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# ========== 3. 加载模型并载入权重 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)
model.load_state_dict(torch.load('stage2_best_model.pth', map_location=device))
model.eval()

# ========== 4. 测试集预测并收集结果 ==========
all_targets = []
all_preds = []
all_images = []  # 存储图像（用于可视化）
all_img_paths = []  # 存储图像路径

with torch.no_grad():
    for inputs, targets, img_paths in tqdm(test_loader, desc='测试中'):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_images.extend(inputs.cpu())  # 存储图像张量
        all_img_paths.extend(img_paths)

# ========== 5. 数值指标输出 ==========
test_acc = accuracy_score(all_targets, all_preds)
print(f"\n测试集准确率: {test_acc:.4f}")
print("\n分类报告:")
print(classification_report(
    all_targets,
    all_preds,
    target_names=test_dataset.classes,
    digits=4
))

# ========== 6. 可视化1：混淆矩阵 ==========
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=test_dataset.classes,
    yticklabels=test_dataset.classes,
    cbar=False
)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# ========== 7. 可视化2：随机展示预测示例 ==========
def denormalize(tensor):
    """反归一化图像，用于显示"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.226])
    tensor = tensor.numpy().transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)
    tensor = tensor * std + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor

# 随机选择8张图像展示
np.random.seed(42)
sample_indices = np.random.choice(len(all_images), 8, replace=False)

plt.figure(figsize=(16, 8))
for i, idx in enumerate(sample_indices):
    img = denormalize(all_images[idx])
    true_label = test_dataset.classes[all_targets[idx]]
    pred_label = test_dataset.classes[all_preds[idx]]
    color = 'green' if true_label == pred_label else 'red'
    
    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.title(f"真实: {true_label}\n预测: {pred_label}", color=color)
    plt.axis('off')

plt.suptitle('测试集预测示例（绿色=正确，红色=错误）', fontsize=16)
plt.tight_layout()
plt.savefig('prediction_examples.png', dpi=300)
plt.show()

# ========== 8. 可视化3：测试集类别分布 ==========
class_counts = np.bincount(all_targets)
plt.figure(figsize=(8, 5))
plt.bar(test_dataset.classes, class_counts, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
plt.xlabel('类别')
plt.ylabel('样本数量')
plt.title('测试集类别分布')
plt.xticks(rotation=15)
for i, count in enumerate(class_counts):
    plt.text(i, count + 1, str(count), ha='center')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300)
plt.show()

# ========== 9. 可视化4：各类别准确率柱状图 ==========
class_acc = []
for cls_idx in range(len(test_dataset.classes)):
    # 计算每个类别的准确率
    cls_mask = np.array(all_targets) == cls_idx
    cls_acc = accuracy_score(np.array(all_targets)[cls_mask], np.array(all_preds)[cls_mask])
    class_acc.append(cls_acc)

plt.figure(figsize=(8, 5))
bars = plt.bar(test_dataset.classes, class_acc, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
plt.xlabel('类别')
plt.ylabel('准确率')
plt.title('各类别准确率')
plt.ylim(0, 1.0)
plt.xticks(rotation=15)
# 在柱上显示数值
for i, acc in enumerate(class_acc):
    plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center')
plt.tight_layout()
plt.savefig('class_accuracy.png', dpi=300)
plt.show()