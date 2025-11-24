# Theta-Gamma 双智能体多跳问答系统

一个基于双智能体架构的多跳问答系统，用于处理复杂的多跳推理问题。系统通过 **Theta Agent** 进行问题规划和答案整合，通过 **Gamma Agent** 进行事实检索和局部回答。

## 📋 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [功能特性](#功能特性)
- [数据集支持](#数据集支持)
- [安装与配置](#安装与配置)
- [使用方法](#使用方法)
- [工作流程](#工作流程)
- [评估指标](#评估指标)
- [输出格式](#输出格式)
- [断点续传](#断点续传)
- [项目结构](#项目结构)

## 🎯 项目概述

本项目实现了一个双智能体协作的多跳问答系统：

- **Theta Agent（规划者）**：负责将复杂问题分解为子问题，调度 Gamma Agent，并整合最终答案
- **Gamma Agent（执行者）**：负责在给定的事实集合中检索相关信息，并回答子问题

系统支持三个经典的多跳问答数据集：2WikiMultihopQA、HotpotQA 和 MuSiQue。

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Pipeline                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Theta Agent (规划者)                             │  │
│  │  1. 问题分解：将复杂问题拆分为子问题              │  │
│  │  2. 调度 Gamma：按顺序调用 Gamma 回答子问题       │  │
│  │  3. 答案整合：基于 Gamma 结果生成最终答案         │  │
│  │  4. 指标计算：计算答案和支持集的六个指标          │  │
│  └──────────────────────────────────────────────────┘  │
│                        ↓                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Gamma Agent (执行者)                             │  │
│  │  1. 事实检索：在给定事实集合中查找相关信息        │  │
│  │  2. 局部回答：基于检索到的事实回答子问题          │  │
│  │  3. 返回结果：返回答案、使用的事实索引和推理      │  │
│  └──────────────────────────────────────────────────┘  │
│                        ↓                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  LLM Client (Ollama API)                         │  │
│  │  封装大语言模型调用，支持重试机制                 │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## ✨ 功能特性

- 🔄 **双智能体协作**：Theta 负责规划，Gamma 负责执行
- 📊 **多数据集支持**：支持 2Wiki、HotpotQA、MuSiQue 三个数据集
- 📈 **实时进度显示**：使用 tqdm 显示处理进度和实时指标
- 💾 **断点续传**：支持中断后从断点继续，避免重复计算
- 📝 **详细日志**：记录完整的推理轨迹和 LLM 调用日志
- 🎯 **六项指标**：同时评估答案质量和支持集质量

## 📚 数据集支持

| 数据集 | 文件 | 描述 |
|--------|------|------|
| **2WikiMultihopQA** | `2wiki_500.json` | 基于维基百科的多跳问答数据集 |
| **HotpotQA** | `hotpotqa_500.json` | 需要多跳推理的问答数据集 |
| **MuSiQue** | `musique_500.json` | 多跳问答数据集，需要多个推理步骤 |

## 🚀 安装与配置

### 环境要求

- Python 3.7+
- 已安装并运行的 Ollama 服务

### 安装依赖

```bash
pip install tqdm requests
```

### 配置环境变量

创建 `.env` 文件（可选，如果不创建则使用默认值）：

```bash
API_URL=http://172.16.120.14:11434/api/generate
MODEL_NAME=deepseek-r1:7b
```

## 📖 使用方法

### 基本用法

运行所有数据集：

```bash
python pipeline.py --data-dir . --log-dir logs
```

### 命令行参数

```bash
python pipeline.py [OPTIONS]

选项:
  --data-dir PATH      数据集文件所在目录（默认：当前目录）
  --datasets LIST      要处理的数据集列表，逗号分隔（默认：2wiki,hotpotqa,musique）
  --limit N           每个数据集的最大样本数（默认：-1，表示全部）
  --log-dir PATH       日志文件保存目录（默认：logs）
```

### 使用示例

**只运行 2Wiki 数据集：**
```bash
python pipeline.py --datasets 2wiki --log-dir logs
```

**限制每个数据集只处理前 10 个样本：**
```bash
python pipeline.py --limit 10 --log-dir logs
```

**指定数据集目录：**
```bash
python pipeline.py --data-dir /path/to/datasets --log-dir logs
```

## 🔄 工作流程

### 1. 问题分解（Theta Agent）

Theta Agent 将原始问题分解为 2-4 个子问题：

```
原始问题: "Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?"

子问题:
  1. Who directed the film 'Move (1970 Film)'?
  2. What is the director's nationality for 'Move (1970 Film)'?
  3. Who directed the film 'Méditerranée (1963 Film)'?
  4. What is the director's nationality for 'Méditerranée (1963 Film)'?
```

### 2. 事实检索与回答（Gamma Agent）

对于每个子问题，Gamma Agent：
- 在给定的事实集合中检索相关信息
- 提取答案（如果事实充分）
- 返回答案、使用的事实索引和推理过程

### 3. 答案整合（Theta Agent）

Theta Agent 基于所有 Gamma 的结果整合最终答案：

```
Gamma 结果:
  - Move 的导演是 Stuart Rosenberg（美国人）
  - Méditerranée 的导演是 Jean-Daniel Pollet（法国人）

最终答案: "no"（他们来自不同国家）
```

### 4. 指标计算

系统计算六个评估指标：
- **答案指标**：EM、F1
- **支持集指标**：EM、F1、Precision、Recall

## 📊 评估指标

系统计算以下六个指标：

### 答案指标（Answer Metrics）

1. **Answer EM (Exact Match)**：预测答案与标准答案完全匹配的比例
2. **Answer F1**：预测答案与标准答案的 F1 分数（基于词级别）

### 支持集指标（Support Metrics）

3. **Support EM**：预测的支持事实集合与标准答案完全匹配的比例
4. **Support F1**：预测的支持事实集合与标准答案的 F1 分数
5. **Support Precision**：预测的支持事实中正确事实的比例
6. **Support Recall**：标准答案中的支持事实被正确预测的比例

## 📄 输出格式

### 日志文件格式

结果保存在 `logs/theta_gamma_{dataset_name}.jsonl` 文件中，每行一个 JSON 对象：

```json
{
  "dataset": "2wiki",
  "example_index": 0,
  "id": "example_id",
  "question": "原始问题",
  "gold_answers": ["标准答案1", "标准答案2"],
  "predicted_answer": "预测答案",
  "answer_em": 1.0,
  "answer_f1": 1.0,
  "support_em": 0.0,
  "support_f1": 0.857,
  "support_precision": 1.0,
  "support_recall": 0.75,
  "theta_gamma_trace": {
    "subquestions": ["子问题1", "子问题2", ...],
    "gamma_results": [
      {
        "step_index": 1,
        "subquestion": "子问题1",
        "gamma_result": {
          "found": true,
          "answer": "答案",
          "selected_fact_indices": [1, 3],
          "selected_fact_texts": ["事实1", "事实2"],
          "reasoning": "推理过程"
        }
      }
    ],
    "theta_final": {
      "answer": "最终答案",
      "reasoning": "整合推理"
    },
    "gamma_call_count": 4,
    "gamma_success_count": 4,
    "predicted_support_indices": [1, 2, 3],
    "gold_support_indices": [0, 1, 2, 3]
  },
  "llm_calls": [...]
}
```

### 控制台输出示例

```
Loading 2wiki from ./2wiki_500.json ...
[2wiki] 发现已处理 277/500 个样本，将从断点继续...
[2wiki]: 55%|██████████████████████████▌| 277/500 [1:48:41<1:27:29, 23.54s/it] ans_EM=0.756 ans_F1=0.823 sup_EM=0.689 sup_F1=0.745

[2wiki] DONE on 500 examples.
  answer_em = 0.7560, answer_f1 = 0.8230
  support_em = 0.6890, support_f1 = 0.7450, support_precision = 0.8120, support_recall = 0.6980
  Results saved to: logs/theta_gamma_2wiki.jsonl
```

## 🔄 断点续传

系统支持断点续传功能，如果程序中断，重新运行会自动从上次停止的地方继续：

### 工作原理

1. **检查已处理样本**：启动时读取日志文件，识别已处理的样本索引
2. **恢复累积指标**：从已处理的样本中恢复六个指标的累积值
3. **跳过已处理样本**：只处理未完成的样本
4. **追加模式写入**：使用追加模式写入日志文件，避免覆盖已有结果

### 使用场景

- 程序意外中断（网络错误、超时等）
- 需要暂停后继续运行
- 分批处理大数据集

### 重新开始

如果需要重新开始处理某个数据集，删除对应的日志文件即可：

```bash
rm logs/theta_gamma_2wiki.jsonl
```

## 📁 项目结构

```
TG/
├── README.md                 # 项目说明文档
├── pipeline.py              # 主流程控制脚本
├── theta.py                 # Theta Agent 实现
├── gamma.py                 # Gamma Agent 和 LLM Client 实现
├── 2wiki_500.json           # 2Wiki 数据集
├── hotpotqa_500.json        # HotpotQA 数据集
├── musique_500.json         # MuSiQue 数据集
├── .env                     # 环境变量配置（可选）
└── logs/                    # 日志输出目录
    ├── theta_gamma_2wiki.jsonl
    ├── theta_gamma_hotpotqa.jsonl
    └── theta_gamma_musique.jsonl
```

## 🔧 核心模块说明

### `pipeline.py`
- 主流程控制脚本
- 数据集加载和管理
- 断点续传功能
- 进度显示和指标统计

### `theta.py`
- **ThetaAgent**：主规划智能体
  - `plan_subquestions()`：问题分解
  - `integrate_answer()`：答案整合
  - `solve_one()`：完整处理流程
  - 六个评估指标的计算

### `gamma.py`
- **LLMClient**：大语言模型客户端
  - 封装 Ollama API 调用
  - 支持重试机制（最多 5 次）
  - 记录所有 LLM 调用日志
- **GammaAgent**：事实检索智能体
  - `build_facts()`：构建事实集合（支持三种数据集格式）
  - `answer_subquestion()`：回答子问题

## 🐛 故障排除

### 网络超时问题

如果遇到网络超时，系统会自动重试（最多 5 次）。如果持续超时，可以：

1. 检查 Ollama 服务是否正常运行
2. 增加超时时间（修改 `gamma.py` 中的 `timeout=120`）
3. 检查网络连接

### 内存不足

对于大数据集，如果遇到内存问题：

1. 使用 `--limit` 参数分批处理
2. 处理完一个数据集后再处理下一个

### 日志文件损坏

如果日志文件损坏导致无法读取：

1. 备份损坏的日志文件
2. 删除损坏的日志文件
3. 重新运行程序（会从断点继续，但会跳过已损坏的记录）

## 📝 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**：确保 Ollama 服务正常运行，并且可以访问配置的 API 地址。

