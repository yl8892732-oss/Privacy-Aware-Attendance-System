# db_manager.py — 人脸考勤系统 · 数据库与隐私管理模块

> **课程：** AIT103 — Introduction to Intelligence Applications  
> **项目：** 智能考勤系统（基于人脸识别 + 活体检测）  
> **角色：** 5 — 数据库与隐私管理（陈锡翘）  
> **技术栈：** Python + SQLite + NumPy + Pandas

---

## 📌 一句话概括

这个文件是整个考勤系统的**"档案室 + 保险箱 + 打卡记录本"**。它负责：存人脸特征、比对是谁、记录打卡时间，并且**保证系统里永远不会出现原始照片**。

---

## 🎯 你在整个系统里是什么位置？

把系统想象成一条工厂流水线：

```
摄像头拍脸 → 活体检测（眨眼/纹理） → 提取人脸特征（512个数字）
                                              ↓
                                    【你在这里】db_manager.py
                                              ↓
                                    存数字 / 比对是谁 / 记打卡 / 删照片
```

你是流水线的**最后一环**。前面的人把"特征数字"传给你，你负责收尾。

---

## 📦 功能清单

| 函数 | 作用 | 大白话 |
|------|------|--------|
| `init_db()` | 初始化数据库 | 搭仓库，建一张表 |
| `register_user()` | 注册新用户 | 把学号和512个数字绑定存起来 |
| `recognize_user()` | 人脸识别 | 拿新数字去库里找最像的那个 |
| `log_attendance()` | 打卡记录 | 把"谁、什么时候、成功没"写进表格 |
| `secure_delete()` | 安全销毁 | 把原始照片从内存里彻底抹掉 |
| `get_user_count()` | 统计人数 | 看看库里注册了多少人 |
| `clear_all_users()` | 清空数据库 | 测试/重置用，慎用 |
| `get_attendance_records()` | 读取打卡记录 | 把CSV日志读出来看看 |

---

## 🔒 核心卖点：隐私保护（Privacy-Aware）

### 为什么安全？

| 普通系统 | 你们的系统 |
|---------|-----------|
| 照片存在服务器云端 | ❌ 照片**不存**，只存数字 |
| 管理员能看到所有人脸 | ❌ 管理员打开数据库只看到**一串浮点数** |
| 注销后照片可能还在备份 | ❌ 注销即删除向量，不留痕迹 |
| 需要联网才能用 | ❌ 本地 SQLite，**断网也能打卡** |

### 技术实现："阅后即焚"

```
摄像头帧（numpy数组，在内存RAM中）
    ↓
提取特征 → 512维向量（仍在RAM中）
    ↓
原始帧 → secure_delete() → 彻底从内存销毁 💥
    ↓
向量 → SQLite本地文件（只有数字，没有图）
    ↓
打卡时 → 只读数字比对，全程不生成图片
```

**答辩金句：**
> "老师请看，我们的数据库里只有学号和二进制数字，没有任何照片文件。原始帧在提取特征后立刻从内存中销毁，实现真正的阅后即焚。"

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install numpy pandas
```

> SQLite 是 Python 内置的，不需要额外安装。

### 2. 直接运行测试

```bash
python db_manager.py
```

如果看到 `All tests passed! Module is ready for integration.`，说明一切正常。

### 3. 查看数据库（答辩展示用）

1. 下载安装 [DB Browser for SQLite](https://sqlitebrowser.org/)
2. 打开生成的 `face_attendance.db` 文件
3. 切换到 `Browse Data` 标签
4. 你会看到 `users` 表，里面只有 `student_id`（学号）和 `embedding`（二进制数字），**没有任何照片**

---

## 📖 接口说明（供角色6集成使用）

### `register_user(student_id, embedding)` → `bool`

**作用：** 注册一个新用户，把学号和特征向量存入数据库。

**参数：**
- `student_id` (str): 学号，例如 `"AIT2509001"`
- `embedding` (np.ndarray): 人脸特征向量，形状 `(512,)`，类型 `float32`

**返回值：**
- `True`：注册成功
- `False`：注册失败（比如输入格式不对）

**示例：**
```python
import numpy as np
from db_manager import register_user

# 模拟一个512维的特征向量（实际由角色2的模型生成）
vector = np.random.randn(512).astype(np.float32)

register_user("AIT2509001", vector)
# 输出: [DB Manager] User 'AIT2509001' registered successfully.
```

---

### `recognize_user(current_embedding, threshold=0.6)` → `str`

**作用：** 拿当前摄像头提取的特征向量，去数据库里找最匹配的人。

**参数：**
- `current_embedding` (np.ndarray): 当前帧提取的特征向量，形状 `(512,)`
- `threshold` (float): 相似度阈值，默认 `0.6`。越高越严格。

**返回值：**
- 匹配的学号字符串（如 `"AIT2509001"`）
- 或者 `"Unknown"`（没有匹配到）

**原理：** 使用**余弦相似度（Cosine Similarity）**计算两个向量有多像。

```
相似度 = 1.0  →  完全一样
相似度 = 0.0  →  完全不像
相似度 ≥ 0.6  →  判定为同一个人
```

**示例：**
```python
from db_manager import recognize_user

result = recognize_user(vector, threshold=0.6)
print(result)  # 输出: AIT2509001 或 Unknown
```

---

### `log_attendance(student_id, status="Success")` → `bool`

**作用：** 记录一次打卡事件到 CSV 文件。

**参数：**
- `student_id` (str): 学号或 `"Unknown"`
- `status` (str): 状态，例如 `"Success"`、`"Failed"`、`"Spoofing_Detected"`

**生成的 CSV 示例：**

| student_id | status | timestamp |
|------------|--------|-----------|
| AIT2509001 | Success | 2026-07-06 08:30:15 |
| AIT2509002 | Success | 2026-07-06 08:31:02 |
| Unknown | Failed | 2026-07-06 08:32:44 |

**示例：**
```python
from db_manager import log_attendance

log_attendance("AIT2509001", status="Success")
# 输出: [DB Manager] Attendance logged: AIT2509001 @ 2026-07-06 08:30:15
```

---

### `secure_delete(var)` → `None`

**作用：** 彻底销毁一个变量，强制释放内存。用于删除原始摄像头帧，防止照片泄露。

**参数：**
- `var`: 要销毁的变量（通常是包含原始图像的 numpy 数组）

**示例：**
```python
from db_manager import secure_delete

frame = camera.capture()  # 原始摄像头帧（包含人脸照片）
embedding = model.extract(frame)  # 提取特征
secure_delete(frame)  # 销毁原始帧，不留痕迹
```

---

## 🏗️ 数据库设计

### 表结构：`users`

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `student_id` | TEXT (PRIMARY KEY) | 学号，唯一标识 |
| `embedding` | BLOB | 人脸特征向量，以二进制存储 |

**为什么只有两列？** 因为系统只存数字，不存照片。`embedding` 列里存的是 512 个 `float32` 数字压缩后的二进制数据，人类看不懂，机器能读。

---

## 📝 报告素材指南

### 第 (3) 节 Design — 画这两张图

1. **隐私数据流图**
   ```
   摄像头帧 → 内存(RAM) → 特征提取 → 向量 → SQLite
                              ↓
                         原始帧 → secure_delete → 销毁
   ```
   说明：数据从进入系统到销毁的全路径，强调"不落地"。

2. **数据库表结构图**
   一张简单的表，两列：`student_id` + `embedding`。

### 第 (4) 节 Implementation — 截这三张图

1. `register_user()` 和 `recognize_user()` 的核心代码截图
2. DB Browser 打开 `face_attendance.db` 的截图，展示里面**没有照片**
3. CSV 打卡日志的截图，展示自动记录功能

### 第 (5) 节 Discussion — 写这三点

1. **隐私对比分析**：存照片 vs 存向量，为什么向量更安全？
2. **性能分析**：SQLite 查询速度多快？支持多少人？
3. **局限性**：向量泄露了能还原成人脸吗？（答：不能，是单向的）

---

## ⚠️ 注意事项

1. **不要改默认数据库路径**（`face_attendance.db`），除非跟角色6商量好。
2. **embedding 必须是 `float32` 类型**，否则 `tobytes()` 和 `frombuffer()` 会出错。
3. **threshold 阈值**默认 0.6，角色2可能会建议调整，以实际测试为准。
4. **CSV 日志是追加写入**，不会覆盖旧记录，方便长期统计。

---

## 🤝 团队协作

| 角色 | 负责人 | 跟你交互的方式 |
|------|--------|--------------|
| 角色2 | 黄晨禹 | 给你 `embedding`（512维向量） |
| 角色6 | 陈星弛 | 调用你的函数，把你的代码拼进主程序 |
| 角色1 | 盘先琰 | 前端展示你的识别结果和打卡状态 |

**你的代码不需要 import 别人的模块**，只接收 `np.ndarray` 类型的向量。这样耦合度最低，最容易集成。

---

## 📅 本周时间线

| 截止时间 | 任务 | 产出 |
|---------|------|------|
| **周五 6.19 晚8点** | 把 `db_manager.py` 推送到 GitHub | GitHub 仓库提交 |
| **周六** | 配合角色6联调，跑通注册→打卡→日志全流程 | 端到端测试通过 |
| **下周** | 截图代码、数据库、CSV，撰写报告段落 | 报告素材 |

---

## 📄 文件清单

- `db_manager.py` — 主代码文件（本文件）
- `face_attendance.db` — 自动生成的 SQLite 数据库（运行后产生）
- `attendance_log.csv` — 自动生成的打卡日志（运行后产生）

---

> **最后提醒：** 这个系统的核心卖点是**隐私**。答辩时打开 DB Browser 给老师看数据库内容，是最有力的证明。
