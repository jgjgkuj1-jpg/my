import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 1. 自定义数据集类 ==========
class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['covid19', 'lungOpacity', 'normal', 'pneumonia']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            # 检查类别文件夹是否存在
            if not os.path.exists(cls_dir):
                raise FileNotFoundError(f"类别文件夹不存在: {cls_dir}")
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.samples.append((img_path, self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # 灰度图转RGB
        if self.transform:
            image = self.transform(image)
        return image, label

# ========== 2. 数据预处理与加载 ==========
# 数据增强（训练集）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一尺寸为ResNet-18输入要求
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(degrees=15),  # 小角度旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度/对比度调整
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.226])  
])

# 验证集（仅标准化，不增强）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.226])
])

# 根据实际文件结构设置路径（E:\COVID19下的dataset）
train_dataset = ChestXRayDataset(root_dir='./dataset/train', transform=train_transform)
val_dataset = ChestXRayDataset(root_dir='./dataset/valid', transform=val_transform)  # 验证集路径

# 数据加载器
batch_size = 4  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ========== 3. 模型构建（ResNet-18迁移学习） ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}（无GPU将使用CPU训练，速度较慢）")

# 加载预训练ResNet-18
model = models.resnet18(pretrained=True)
# 修改输出层为4类
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失

# ========== 4. 分阶段训练函数 ==========
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()  # 训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc='训练中'):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # 统计指标
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(loader)
    avg_acc = correct / total
    return avg_loss, avg_acc

def val_epoch(model, loader, criterion, device):
    model.eval()  # 验证模式
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算
        for inputs, targets in tqdm(loader, desc='验证中'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(loader)
    avg_acc = correct / total
    return avg_loss, avg_acc

# ========== 5. 第一阶段训练：仅训练分类头（冻结卷积层） ==========
print("\n===== 第一阶段训练：冻结卷积层，训练分类头 =====")
# 冻结所有卷积层参数
for param in model.parameters():
    param.requires_grad = False
# 解冻分类头
for param in model.fc.parameters():
    param.requires_grad = True

# 优化器与调度器
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# 训练参数
epochs_stage1 = 10
best_acc = 0.0
patience = 5  # 早停耐心值
counter = 0

# 记录训练曲线
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(epochs_stage1):
    print(f"\nEpoch {epoch+1}/{epochs_stage1}")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
    
    # 保存指标
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # 打印结果
    print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
    print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
    
    # 学习率调度
    scheduler.step(val_acc)
    
    # 早停与最佳模型保存
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'stage1_best_model.pth')
        print(f"保存新的最佳模型（验证准确率: {best_acc:.4f}）")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"早停触发（连续{patience}个epoch未提升），第一阶段训练结束")
            break

# ========== 6. 第二阶段训练：微调深层卷积层 ==========
print("\n===== 第二阶段训练：微调深层卷积层 =====")
# 加载第一阶段最佳模型
model.load_state_dict(torch.load('stage1_best_model.pth'))

# 解冻深层卷积层（layer3和layer4）
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# 优化器与调度器（学习率降低）
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# 训练参数
epochs_stage2 = 10
counter = 0  

for epoch in range(epochs_stage2):
    print(f"\nEpoch {epoch+1}/{epochs_stage2}")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
    
    # 打印结果
    print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
    print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
    
    # 学习率调度
    scheduler.step(val_acc)
    
    # 早停与最佳模型保存
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'stage2_best_model.pth')
        print(f"保存新的最佳模型（验证准确率: {best_acc:.4f}）")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"早停触发（连续{patience}个epoch未提升），第二阶段训练结束")
            break

# ========== 7. 训练曲线可视化 ==========
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失', color='blue')
plt.plot(val_losses, label='验证损失', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('损失曲线')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='训练准确率', color='blue')
plt.plot(val_accs, label='验证准确率', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('准确率曲线')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
print("\n训练曲线已保存为 'training_curves.png'")
plt.show()