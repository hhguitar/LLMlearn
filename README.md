# Tesla Cross-Year Financial Report QA System

## 1. 环境管理

本项目使用 conda + pip 进行环境管理，前端使用Streamlit。

### 推荐环境

- Python 3.10 / 3.11
- conda
- pip

### 安装步骤

```bash
conda env create -f environment.yml
conda activate tesla-qa
```

如需补装部分依赖，可使用：

```bash
pip install -r requirements.txt
```

### 环境变量

项目支持通过 `.env` 或系统环境变量配置模型与下载参数。

示例：

```env
SEC_USER_AGENT=YourName your_email@example.com
LLM_API_KEY=your_dashscope_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
EMBEDDING_BACKEND=local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

说明：

- `SEC_USER_AGENT`：下载 SEC 文件时使用，需为英文/ASCII 字符。
- `LLM_API_KEY`：阿里云百炼 / DashScope API Key。
- `LLM_BASE_URL`：DashScope OpenAI 兼容接口地址。
- `LLM_MODEL`：当前默认使用 `qwen-plus`。
- `EMBEDDING_BACKEND`：支持本地 embedding。
- `EMBEDDING_MODEL`：当前默认使用 `BAAI/bge-small-en-v1.5`。

---

## 2. 数据概览

### 2.1 数据来源

数据来源为 Tesla 官方 SEC 申报文件，覆盖：

- 10-K 年报
- 10-Q 季报

### 2.2 实际处理范围

当前系统设计目标覆盖以下年份：

- 2021
- 2022
- 2023
- 2024
- 2025（如当年文件可获取）

### 2.3 已处理文档类型

- HTML 版本主文档（优先）
- PDF 财报（作为补充解析来源）

### 2.4 当前实现状态

当前版本已经具备以下能力：

- Tesla filings 列表抓取
- 文档下载
- HTML / PDF 解析
- 文本块和表格块生成
- 向量索引和 BM25 索引构建
- Streamlit 问答界面
- Qwen / DashScope 问答调用
- 检索证据展示
- 高阶测试集与失败案例分析

---

## 3. 系统设计抉择

### 3.1 文档解析策略

#### 目标
从 Tesla 财报中同时提取：

- 文本段落
- 财务表格
- 元数据（年份、季度、文件、报告期、章节、页码等）

#### 当前实现
- 优先解析 HTML filing
- 补充解析 PDF
- 表格转换为：
  - Markdown
  - JSON rows
- 为每个文本块和表格块保留 metadata

#### 设计原因
复杂财务问答不能只依赖纯文本。很多关键数据（毛利率、研发费用、现金流）都存在于结构化表格中，因此必须把表格单独抽出并保留来源信息。

---

### 3.2 分块策略

#### 当前采用的策略
- 文本按章节/段落进行切分
- 表格作为独立完整单元，不再按长度切开
- 每个 chunk 保留：
  - form type
  - filing date
  - report date
  - source file
  - section title
  - chunk type
  - table title / page info

#### 设计原因
对于财报问题，单纯按 token 长度切块容易破坏逻辑结构，尤其会导致：
- “管理层讨论”与对应财务数据无法关联
- 表格被拆碎后失去可读性
- 年份/季度与指标错位

因此本项目采用 **结构感知分块**：
- 文本依赖原始章节
- 表格保证完整性
- 后续检索时利用 metadata 进行过滤和对齐

---

### 3.3 向量化与索引

#### 向量化模型
- `BAAI/bge-small-en-v1.5`

#### 检索索引
- 向量索引：ChromaDB
- 关键词索引：BM25

#### 设计原因
财报问答既需要语义召回，也需要精确术语命中。例如：
- `Free Cash Flow`
- `2022 Q3`
- `Automotive gross margin`
- `Management's Discussion and Analysis`

仅使用向量检索容易漏掉年份、财务术语和表格标题；仅使用 BM25 又容易在跨文档语义问题上召回不足。因此采用 **BM25 + 向量混合检索**。

---

### 3.4 生成模型

#### 当前采用模型
- `qwen-plus`（DashScope OpenAI 兼容接口）

#### 调用方式
- API 调用
- 使用 OpenAI 兼容 SDK 接口

#### 设计原因
最初版本使用 OpenAI 接口，但实际运行中受 API quota 限制。后续切换到 Qwen / DashScope 后，问答链路可以稳定运行，因此当前项目默认使用 Qwen 作为生成模型。

---

### 3.5 复杂问题回答流程

当前系统采用如下流程：

1. 问题输入
2. 检索阶段
   - 文本块检索
   - 表格块检索
3. 结果融合
   - 合并 BM25 与向量召回结果
4. 答案生成
   - 将证据块送入 Qwen
5. 结果展示
   - 输出答案
   - 输出引用证据
   - 输出检索调试信息

#### 当前局限
当前流程能够处理基础跨文档问答，但对于更复杂的问题，例如：
- 先定位季度
- 再计算数值
- 再提取对应 MD&A

目前还不具备成熟的 agent 化多步执行器，因此复杂题表现不稳定。

---

## 4. 运行方式

### 4.1 构建索引

```bash
python scripts/run_pipeline.py
```

流程包括：

- 下载 filings
- 解析文档
- 分块
- embedding
- 建立 Chroma / BM25 索引

### 4.2 命令行问答

```bash
python scripts/ask.py "Compare Tesla's automotive gross margin by quarter from 2021 to 2023 and identify the highest quarter."
```

### 4.3 Streamlit 页面

```bash
streamlit run app/streamlit_app.py
```

界面功能包括：

- 选择文档范围（仅 10-K / 仅 10-Q / 所有文档）
- 输入问题
- 展示答案
- 展示引用证据
- 展示检索调试信息

---

## 5. 高阶测试集与结果摘要

为评估系统在复杂财务问答上的能力，构建了如下测试集。

| 编号 | 测试问题 | 能力类型 | 结果 |
|---|---|---|---|
| Q1 | 对比 2021–2023 年哪个季度汽车毛利率最高，并提取当时 MD&A 宏观背景 | 文本+表格关联 / 跨季度比较 | 部分成功 |
| Q2 | 计算 2022 年四个季度研发费用总和，并与 2021 年全年对比 | 数值计算 / 跨文档汇总 | 失败 |
| Q3 | 比较 2021 年 10-K 与 2023 年 10-K 中中国市场风险的描述变化 | 跨文档文本对比 | 部分成功 |
| Q4 | 描述 2021–2023 年自由现金流季度波动，并指出第一次明显转弱 | 时间顺序 / 数值推理 | 失败 |
| Q5 | 找到汽车毛利率最低的季度，并提取该季度 MD&A 的关键句子 | 文本+表格关联 | 部分成功 |
| Q6 | 2022 年哪份季报首次提到供应链挑战？该季度营收环比变化如何？ | 多步推理 / 文本+数值 | 失败 |
| Q7 | 对比 2022 与 2023 各季度营业利润率和汽车毛利率变化 | 跨季度比较 | 失败 |
| Q8 | 哪份文件首次提到某工厂产能瓶颈？后续如何演变？ | 时间顺序 / 跨文档追踪 | 失败 |

### 结果总结
当前系统已经能完成基础检索增强问答，但在以下问题类型上仍较弱：

- 跨季度完整覆盖不足
- 数值计算不稳定
- 时间顺序“首次出现”判定弱
- 文本与表格之间的季度绑定不足

---

## 6. 失败案例深度剖析

详见单独文档：

- `docs/FAILURE_ANALYSIS.md`

其中详细分析了至少 5 个典型失败案例，涵盖：

- 失败表象
- 溯源排查
- 根本原因
- 改进方案

---

## 7. 当前系统能力边界

### 已解决的问题
- 支持跨年 filings 下载与解析
- 支持文本块和表格块联合索引
- 支持混合检索
- 支持 Qwen 生成答案
- 支持 Streamlit 展示答案和证据

### 仍未完全解决的问题
- 季度级财务指标提取不系统
- 程序化数值计算不足
- 时间线问题不稳定
- “先找季度、再找同季度文本”两阶段推理不足
- 中文财务术语到英文财报术语的 query rewrite 不足

---

## 8. 后续改进方向

1. 增加 query rewrite
   - 将中文财务问题扩展为英文财报术语
2. 加强年份 / 季度硬过滤
   - 避免 2020 文档被召回到 2021–2023 问题中
3. 构建季度级结构化指标表
   - automotive gross margin
   - R&D
   - operating cash flow
   - capex
4. 引入 deterministic calculato
   - 求和
   - 排序
   - 环比 / 同比
5. 设计两阶段 Agent 流程
   - 阶段 1：找季度/数值
   - 阶段 2：找对应解释性文本
6. 对表格建立主题索引
   - 毛利率表
   - 现金流量表
   - 研发费用表
   - 债务表
7. 增加时间线检索模式
   - 解决“首次提到”“第一次转弱”类问题

---

## 9. 项目目录建议

```text
tesla_qa_full/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── index/
├── docs/
│   └── FAILURE_ANALYSIS.md
├── evals/
│   ├── test_questions.md
│   └── test_questions.jsonl
├── scripts/
│   ├── run_pipeline.py
│   ├── ask.py
│   └── run_eval.py
├── src/
│   └── tesla_qa/
│       ├── parser.py
│       ├── chunking.py
│       ├── indexer.py
│       ├── retriever.py
│       ├── llm.py
│       ├── qa_pipeline.py
│       └── ...
├── .env.example
├── environment.yml
└── README.md
```

---

