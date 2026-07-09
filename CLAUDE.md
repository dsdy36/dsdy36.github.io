# FS_generater — TAFFIES 自动化评分系统

## 项目结构

```
FURP/
├── FS_generater-v1/           ← TAFFIES 管线（15 个 .py 文件）
│   ├── taffies_fs_generator.py  → 核心：TAFFIES FS 生成器（check function, ~1100 行）
│   ├── test_fs_generator.py     → 测试驱动 FS 生成器（语义方法，实验性）
│   ├── plan_d.py                → 编排器（Phase 0/1 + 后处理）
│   ├── ai_pipeline.py           → DeepSeek API、数据收集、Phase 0/1
│   ├── ground_truth.py          → README 解析（Error/Mistake 分离）
│   ├── coverage.py              → FCC 覆盖检查
│   ├── blind_test.py            → 盲测（支持 check function + test_case）
│   ├── round5_refine.py         → check function 迭代精炼（最多 5 轮/FS）
│   ├── round4_refine.py         → regex 正反例精炼（legacy，regex 时代用）
│   ├── ast_diff_fs.py           → AST 差分规则合成（确定性方法，实验性）
│   ├── sandbox.py               → reference 验证（Flask test_client）
│   ├── audit.py                 → 最终审计
│   ├── config_generator.py      → 提交扫描
│   ├── comment_stripper.py      → 注释过滤
│   ├── verify_readmes.py        → README 可信度验证
│   ├── analyze_fp.py            → FP 分布分析
│   ├── audit_check_fns.py       → check function 质量审计
│   ├── check_context_patterns.py → 上下文敏感 pattern 分析
│   ├── find_example.py          → TP/FP 代码示例查找
│   ├── question/code/           → iMusic.py 模板（含 stub 函数）
│   ├── submission/              → 50 个 CW 学生提交
│   │   └── Sxxx/ → task1.py + task2.py + task3.py + README.md
│   ├── references/              → AI 生成的 reference 函数
│   └── output/q1_iMusic/        → fs_registry.json + 各种中间输出
│
├── CW-generater/               ← 学生提交生成器
│   ├── main_v5.py               → 入口: python main_v5.py -n 50
│   ├── pattern_matrix.py        → rubric → patterns × variants（硬编码 + AI 回退）
│   ├── generate_readme.py       → 生成 README.md
│   ├── convert_for_v1.py        → CW 格式 → V1 submission 格式
│   ├── inject_sql_patterns.py   → SQL 注入后处理（注入 %/.format 变体）
│   ├── fix_vague_labels.py      → AI 批量重写模糊 CW label
│   ├── check_readme_quality.py  → README 质量检查
│   ├── check_remaining.py       → 残余模糊 label 检查
│   ├── test_phase0.py           → Phase 0 测试
│   └── submissions_imusic_v5/   → 生成的学生提交
│
└── test/                        ← 手写 AFS（教学参考）
```

## 完整流程

```bash
# === CW 端：生成训练数据 ===
cd CW-generater

# 1. 生成 50 个学生提交（AI 生成代码 + 确定性注入 pattern variant）
python main_v5.py -n 50

# 2. 生成 README（记录每个学生注入了哪些 good/bad pattern）
python generate_readme.py submissions_imusic_v5 50

# 3. 格式转换（CW 格式 → V1 per-task 格式）
python convert_for_v1.py submissions_imusic_v5 ../FS_generater-v1/submission

# === FS 端：生成 Feedback Signature ===
cd ../FS_generater-v1

# 4. TAFFIES FS 生成（基于 README label 聚类 → AI 生成 check function）
python taffies_fs_generator.py question submission q1_iMusic

# 5. 迭代精炼（正反例驱动收紧 check function）
python round5_refine.py output/q1_iMusic/fs_registry_taffies.json submission

# 6. 盲测验证
python blind_test.py output/q1_iMusic/fs_registry_taffies.json submission

# 7. FP 分析
python analyze_fp.py
```

### 管线内部流程

```
Phase 0:  AI 从 PDF 提取 rubric → rubric_cache.json（缓存，仅首次需要 API）
Phase 1:  AI 生成 reference 函数片段 → 沙箱验证

TAFFIES FS 生成:
  A. 聚类: 从 50 个 README 中读取每个学生注入了哪个具体 pattern variant
     → 相同 pattern 的学生归为一组（如 "pandas" 组、"readlines" 组）
  B. 生成: 每个 cluster → AI 生成 1 条 check function（Type B: 检测 PRESENCE）
     → System prompt 禁止 Type A（(?!...), return not, "not in"）
  C. FCC:  迭代补缺（最多 3 轮）
     → 优先补负向 FS（用 rubric bad_patterns 作为 label）
     → 每个 gap 学生最多生成 2 条负向 FS
  D. 验证: 负向 FS 匹配 reference → 降权 weight=0.5（不删除）
           正向 FS 匹配 template → 降权 weight=0.5

后处理:
  - Criteria Filter: 无 bad pattern 的 criterion → 删除该 criterion 所有负向 FS
  - Broad Filter: 匹配率 >40% 且 FPR >50% → 降权（GT-aware）
  - Variable Gen: tokenize 学生代码 → 替换非白名单变量名为 \w+

输出: fs_registry.json（check function 格式）
```

## 核心概念

- **FS (Feedback Signature)** = `check_function` (Python `def check(code) -> bool`) + `feedback` (学生可读文本)。**签名格式从 regex 进化为 check function。**
- **Check function**: AI 生成的 Python 函数，检测代码中是否存在特定模式。支持 `ast.parse/walk`、`re.search`、字符串方法。存储在 FS dict 的 `check_function` 字段，`signature_type = "check_function"`。
- **Type B only**: 只检测"存在某模式"（PRESENCE）。System prompt 明确禁止 Type A（`(?!...)`、`return not`、`"not in"`）。
- **Type A vs Type B**:
  - Type A: 检测"缺失某好模式"（如 `def func:(?!.*csv\.)`）→ 精度低，容易误匹配正确代码
  - Type B: 检测"存在某坏模式"（如 `import pandas`）→ 精度高，只匹配真正写错的学生
- **FCC (Feedback Coverage Checker)**: 每个学生每个 criterion 至少 1 个 FS 匹配。用于迭代补充。
- **Ground Truth**: CW 生成的 README.md 精确记录每个学生注入了哪些 pattern variant。
- **14 个评分标准**: Task1: RQ1_1-RQ1_4, Task2: RQ2_1-RQ2_4, Task3: RQ3_1-RQ3_6。
- **Error vs Mistake**:
  - ❌ Error pattern: 一定是错的（f-string SQL 注入、pandas、硬编码路径）→ 正常生成负向 FS
  - ⚠️ Mistake to include: 可能出现在正确代码中（INSERT OR IGNORE、SELECT COUNT before INSERT）→ 生成负向 FS 但 prompt 告知 AI "只在缺少伴随检查时标记"

---

## 所有方案结果汇总

实验条件: 50 个 CW 生成的学生，14 个 criterion。

| 方案 | P | R | F1 | 负向 FS | 核心问题 |
|------|:--:|:--:|:--:|:--:|------|
| **1. 旧 Plan D (regex)** | 0.18 | 0.90 | 0.30 | 89 | 大量 Type A 过度匹配，447 FP |
| **2. TAFFIES check fn (语法)** | 0.29 | 0.37 | 0.29 | 17 | Type B 改善但 AI 质量方差大 |
| **3. TAFFIES + FPR 过滤** | **1.00** | 0.05 | 0.10 | **3** | 仅 3 条 FS 达到 P=1.0 |
| 4. Round 5 迭代精炼 | 0.11 | 0.37 | 0.17 | 18 | AI 无法收紧而不失 TP |
| 5. AST Diff 规则合成 | — | — | — | 0 | 无区分性语法特征 |
| 6. 测试驱动 v1 (简陋沙箱) | 0.28 | 0.28 | 0.28 | 68 | 沙箱 mock 环境不准确 |
| 7. 测试驱动 v2 (Flask) | 0.29 | 0.19 | 0.23 | 70 | 沙箱不稳定，5 个 criterion 全败 |
| **8. 沙箱先行 (subprocess 隔离)** | 0.146 | 0.390 | 0.213 | 14 | 沙箱行为测试可靠(200/200 成功)，但 AI 生成 check function 时 3/4 行为负向簇 FAILED。行为数据无法自动转化为语法特征 |

### 3 条完美 FS（方案 3）

```
RQ1_3: try_except_integrity_error       — 检测 try/except IntegrityError (TP=5, FP=0)
RQ1_3: try_except_integrity_error_for   — 检测另一变体 (TP=5, FP=0)
RQ1_4: bare_return_without_value        — 检测 return 没有返回值 (TP=2, FP=0)
```

**共同特征**: 都是强信号 pattern——代码中存在独一无二的语法结构明确标识了错误。

### FP 分布特征（方案 2，17 条 FS）

```
6 条精确 FS (FPR=0%~40%):  TP=15,  FP=2,  P=0.88  ← 可用
11 条宽 FS (FPR>40%):       TP=118, FP=327, P=0.27  ← 不可用
88% 的 FP 来自 9 条 FS
```

---

## 已解决的所有问题

| # | 问题 | 解决方案 | 文件 |
|---|------|---------|------|
| 1 | Type A 负向 FS 导致 447 FP | System prompt 禁止 Type A，只生成 Type B check function | taffies_fs_generator.py |
| 2 | AI 输出缩进错误 | AI 输出完整 `def check(code):` 函数，不做包裹 | taffies_fs_generator.py |
| 3 | 模板代码混入学生代码 | `_extract_student_functions()` 只提取学生写的函数体 | taffies_fs_generator.py |
| 4 | 模板函数名硬编码 | `_get_template_context()` 动态从 question/code/iMusic.py 提取 | taffies_fs_generator.py |
| 5 | Error/Mistake 标签混淆 | `ground_truth.py` 分离 `bad` 和 `mistake` 列表 | ground_truth.py |
| 6 | CW label 模糊 ("deliberately implement incorrectly") | pattern_matrix AI fallback + 硬编码全部改为 Type B 描述 | CW-generater/pattern_matrix.py |
| 7 | CW label AI fallback 含未转义单引号 | `fix_vague_labels.py` 的 `apply_rewrites` 加 `replace("'", "\\'")` | CW-generater/fix_vague_labels.py |
| 8 | FCC 只补正向不补负向 | FCC 优先补负向 + 用 rubric bad_patterns 驱动 label | taffies_fs_generator.py |
| 9 | 盲测不支持 check function | 加 `_fs_matches()` 同时支持 regex 和 check function | blind_test.py |
| 10 | CW API 调用无缓存 | `main_v5.py` 加 `variant_cache.json` 路径 | CW-generater/main_v5.py |
| 11 | 负向 FS 匹配 reference 被直接删除 | 改为降权 weight=0.5（对齐旧系统策略） | taffies_fs_generator.py |
| 12 | `main_v5.py` 无 Python 输出缓冲 | 加 `-u` flag | 运行命令 |
| 13 | pattern_matrix 硬编码 6 处 Type A check_regex (`(?!...)`) | 全部改为 Type B（检测具体存在的代码模式） | CW-generater/pattern_matrix.py |
| 14 | pattern_matrix 好模式错标为坏（INSERT OR IGNORE 被标为 bad） | 保留但标注为 alternative approach | CW-generater/pattern_matrix.py |
| 15 | round5_refine 全量函数发给 AI | 改为只发匹配的代码片段（最多 8 行，最多 12 个） | round5_refine.py |
| 16 | round5_refine 只做 1 轮 AI 调用 | 改为最多 5 轮迭代，每轮重新验证 TP/FP | round5_refine.py |
| 17 | 裸 `os.listdir(submission)` 受旧数据污染 | 添加 `Remove-Item submission -Recurse -Force` 后重新复制 | 运行流程 |
| 18 | 旧 `submission_backup/` + `submission_backup_old/` 残留 | 删除 + 删除 20 个无用 .py 文件 | 文件清理 |
| 19 | 沙箱需要 Docker 但环境无 Docker/Podman | subprocess + temp file 隔离方案，进程级隔离足够 CW 生成代码使用 | runtime/subprocess_executor.py |
| 20 | RQ1_3 行为测试 `inspect.getsource()` 在子进程返回空 | 44/50 学生返回 UNKNOWN。需改用代码文本搜索替代 inspect | runtime/test_generator.py (待修复) |

---

## 未解决的根本问题

### 问题 1: 语法方法的上限 (P=1.0 但 R=0.05)

14 个 criterion 中，只有 RQ1_3 和 RQ1_4 有 precise FS。其余 12 个 criterion 的 check function 均存在 FP，已经过 4 种精炼方法（Round 5 迭代、AST diff、FPR 过滤、重跑管线）都无法消除。

**根因**: 不存在语法层面的区分特征。TP 和 FP 代码在该 criterion 相关的代码中有着完全相同的 AST 结构和字符串模式。区别在语义层面（"SELECT COUNT 是否在 INSERT 之前执行"），而语义不在代码文本中。

**具体分析**: 对 14 个 criterion 逐一分析，13 个不可用单一语法特征分离（见下文"上下文敏感分析"）。

### 问题 2: AI 精炼的失效

Round 5 迭代精炼最多 5 轮：8 条候选 FS，0 条达到 FPR=0%，3 条有所改善（FP 下降 20-30%），5 条无改善。

**根因**: AI 收到 TP/FP 代码片段后，被要求"找到区分特征并重写 check function"。但区分特征不存在于代码语法中——存在的是语义差异（代码执行的意图和上下文），AI 无法从语法片段中提取语义差异。

### 问题 3: 22 个 CW label 中 14 个仍含 Type A 语言

AI 批量重写后，14 个 label 仍含 "without" 或 "not"。这些 label 被 AI 用于生成 check function，AI 会根据"不要做 X"的描述写出"检测 X 存在"的函数——但检测到的 X 往往在正确代码中也存在。

### 问题 4: 测试驱动的三个工程难题

| 难题 | 描述 | 为什么难 |
|------|------|---------|
| **语义鸿沟** | 从 rubric 描述到可执行测试的转换 | "确保 commit 事务"→ 如何测试？执行代码后数据已被写入，无法区分是否有 commit 还是未关闭连接 |
| **沙箱不稳定性** | Flask 应用组装和执行环境 | 学生代码有 import 依赖、全局变量、硬编码路径、无限循环——沙箱 5/14 criterion 全败 |
| **测试充分性** | 测试可能"假通过" | 传入合法参数时正确和错误代码都通过，只有传入恶意参数才能区分——AI 难以自动生成针对性的恶意输入 |

### 问题 5: 语义测试的理论局限

即使沙箱完美运行，也**无法覆盖所有 criterion**：

- RQ1_1 "使用 csv 模块解析" → 无法测试——两种方法都产生相同的数据库输出
- RQ1_2 "关闭数据库连接" → 无法在 Python 中可靠检测——连接可能在 GC 时隐式关闭
- RQ2_4 "使用 flash 消息提示错误" → flash() 无返回值，无法断言"是否调用了正确的 flash"
- RQ3_1 "按名称排序" → 检查输出列表的排序结果可以测试（✅ 可行）

**可测试的 criterion**: RQ1_3, RQ2_1, RQ2_3, RQ3_1, RQ3_4
**不可测试的 criterion**: RQ1_1, RQ1_2, RQ1_4, RQ2_2, RQ2_4, RQ3_2, RQ3_3, RQ3_5, RQ3_6

---

## 上下文敏感 pattern 分析

对 50 个学生的所有 criterion 进行分析，检查是否存在能区分 TP 和 FP 的语法特征（AST 签名 + 字符串模式）：

| Criterion | 可分离 | 说明 |
|-----------|:--:|------|
| RQ1_1 | ⚠️ 弱可分离 | `no_parameterized` 在 TP 中出现更多（83% vs 51%） |
| RQ1_2-RQ3_6 | ❌ 不可分离 | 所有特征在 TP 和 FP 中分布几乎相同 |

示例——RQ2_3 "ORDER BY 验证":
```
has_fstring_order: TP=74%, FP=75%  ← 几乎完全重叠
has_whitelist:     TP=26%, FP=21%  ← 即使有 whitelist 也只差 5%
```
没有单一语法特征能区分"有验证的正确代码"和"无验证的错误代码"。

---

## 环境

- Python: `C:\ProgramData\anaconda3\python.exe`
- DeepSeek API: `.env` 在 `FS_generater-v1/` 和 `CW-generater/`
- DeepSeek Model: `deepseek-chat`（`deepseek-reasoner` 可选，需处理 system→user message 转换）
- 依赖: `openai`, `PyPDF2`, `pyyaml`, `python-dotenv`, `flask`

---

## 文献出处总览

### 本项目中已使用的方法及其文献来源

| 文献 | 年份 | 在本项目中的应用 | 文件/模块 |
|------|:--:|------|------|
| **TAFFIES** — Tailored Automated Feedback Framework (原始方法论) | — | FS = signature + feedback 配对；criterion-by-criterion 聚类；FCC 覆盖循环；Type A/B 区分；reference 验证 | `taffies_fs_generator.py`, CLAUDE.md §核心概念 |
| **OverCode** (Glassman et al.) | 2015 | 学生代码变体可视化与聚类——启发 `cluster_by_pattern()` 将相同 pattern variant 的学生归为一组 | `taffies_fs_generator.py:cluster_by_pattern()` |
| **GumTree** (Falleri et al.) | 2014 | AST 级代码差分算法——启发 `ast_diff_fs.py` 尝试从 TP/FP 代码中提取区分性 AST 子树 | `ast_diff_fs.py` |
| **Execution-Guided Neural Program Synthesis** (Chen et al.) | 2019 | 执行结果反馈指导代码生成——启发行为沙箱方案 (Phase 1.5)：先用 subprocess 执行学生代码观察行为,再用行为标签指导 FS 生成 | `runtime/test_generator.py`, `runtime/batch_runner.py` |
| **Test-Driven Repair** (Long & Rinard) | 2016 | 测试驱动代码修复——启发 `round5_refine.py` 的正反例迭代精炼: 用 TP/FP 代码片段作为"测试"来验证 check function 是否正确 | `round5_refine.py` |
| **DECKARD** (Jiang et al.) | 2007 | AST 子树相似度聚类检测代码克隆——启发 FCC 覆盖检查中对相似学生代码的分组 | `coverage.py` |
| **code2vec** (Alon et al.) | 2019 | 代码 → 向量嵌入——路径 3 的备选方案: 将学生代码转为向量后用分类器预测是否含错误模式 | (备选,未实现) |
| **ASTNN** (Zhang et al.) | 2019 | AST 切片 → RNN → 代码分类——路径 3 的深度学习方案,需更多数据 | (备选,未实现) |
| **CodeBERT** (Feng et al.) | 2020 | 预训练代码表示——可用于代码嵌入 + 分类 | (备选,未实现) |

### 本项目中引用的综述与评测文献

| 文献 | 年份 | 内容 |
|------|:--:|------|
| **Keuning et al.** — "A Systematic Literature Review of Automated Feedback Generation for Programming Exercises" | 2018 | 自动反馈生成的系统文献回顾,覆盖 101 篇论文。建立了反馈类型分类法(知识型/语法型/语义型/策略型),确认了"语法特征的区分性不够"是普遍问题 |
| **Messer et al.** — "Automated Feedback for Introductory Programming" | 2024 | 最新综述,聚焦 LLM-based feedback。确认 LLM 在生成语法检查方面表现良好,但在理解代码意图/行为方面仍有根本困难 |
| **LLM-as-Judge 系列** (OpenAI, Stanford, 各大学) | 2024-2025 | 用 LLM 直接评估代码质量——taffies-2026 GradeForge 的 AI review pass 即属此范式 |

### 对比模式挖掘与程序合成文献(新方案引用)

| 文献 | 年份 | 与本方案的相关点 |
|------|:--:|------|
| **GumTree** (Falleri et al.) | 2014 | AST 树匹配与差分——对比挖掘的核心: 给定两群代码(VULNERABLE vs SAFE),找区分性 AST 子树 |
| **DECKARD** (Jiang et al.) | 2007 | 子树向量化 + LSH 聚类——用于在大量学生代码中高效找到相似模式 |
| **Execution-Guided Synthesis** (Chen et al.) | 2019 | 执行 → 反馈 → 重新生成——行为沙箱的核心范式: 不信任代码文本,验证代码行为 |
| **Program Synthesis by Sketching** (Solar-Lezama) | 2008 | 测试驱动的程序合成——给定 oracle(行为标签),合成满足约束的程序(check function) |
| **Test-Driven Repair** (Long & Rinard) | 2016 | 用测试用例指导代码修复——直接对应: 用行为标签(ground truth)作为测试,验证 check function |
| **Contrastive Code Representation Learning** (Jain et al.) | 2021 | 对比学习在代码中的应用——训练模型区分"正确"和"错误"代码的嵌入表示 |

---

## 方案 8: CW 确定性注入 + 对比挖掘 + 行为沙箱混合方案

### 背景: 为什么需要这个方案

方案 7（沙箱先行）证明了:
- **行为沙箱本身可靠**: 200 个测试 100% 完成,零超时,VULNERABLE/SAFE 分离清晰
- **但 AI 无法利用行为信号生成 check function**: 4 个行为负向簇中 3 个 FAILED(AI 生成的 check function 在 representative 代码上返回 False),和 CLAUDE.md 问题 2 一致

根因: 行为标签告诉我们**哪些学生错了**,但不能直接告诉 AI **用什么语法特征检测这个错误**。两个学生可能有 AST 结构完全相同的代码(都用 f-string ORDER BY),但一个因为前置验证而行为安全,另一个没有验证而行为危险。AI 无法从代码文本中区分这种差异。

### 方案核心: 三步流水线

```
Step 1 (CW端): 确定性模式注入
  对每个 criterion 定义一对确定性模板(vulnerable + safe)
  AI 生成代码后,后处理器用正则匹配+替换确定性地注入关键模式
  → 干净对比对: vulnerable/safe 代码只在关键 pattern 上有差异

Step 2 (FS端): 对比特征挖掘
  对 VULNERABLE 群和 SAFE 群的代码做统计对比
  Token n-gram / AST 子树的频率差异
  → 区分性特征列表(token patterns, AST patterns)

Step 3 (FS端): AI 基于特征组装 check function
  将区分性特征列表喂给 AI
  AI 的任务从"自己发现特征"降级为"根据已有特征组装 check function"
  → 预期质量大幅提升
```

### 确定性注入设计

为每个 criterion 在 CW-generater 中新增 `pattern_injector.py`,在 AI 生成代码之后运行:

```python
DETERMINISTIC_INJECTIONS = {
    # ── RQ2_3: ORDER BY 验证 ──
    "RQ2_3": {
        "vulnerable": {
            # 在 get_statistics() 中注入无验证的 f-string ORDER BY
            "target_func": "get_statistics",
            "inject": 'query += f" ORDER BY {sort_column} {sort_order}"',
            "ensure_absent": ["allowed", "ALLOWED", "valid_sort", "whitelist",
                              "if sort_col", "if sort_by not in"],
        },
        "safe": {
            # 注入 allowlist 验证
            "target_func": "get_statistics",
            "inject": '''
    allowed = {"NumberOfTracks": "NumberOfTracks", "Duration": "Duration",
               "TotalCost": "TotalCost", "AverageCost": "AverageCost",
               "PlaylistName": "p.Name"}
    sort_col = allowed.get(sort_column, "p.Name")
    sort_ord = sort_order if sort_order in ("ASC", "DESC") else "ASC"
    query += f" ORDER BY {sort_col} {sort_ord}"
''',
        },
    },
    # ── RQ2_1: 参数化查询 ──
    "RQ2_1": {
        "vulnerable": {
            "target_func": "get_all_genres",
            "inject": 'cursor.execute(f"SELECT GenreId, Name FROM Genre ORDER BY Name")',
            "ensure_absent": ["cursor.execute(?,", "cursor.execute('?',"],
        },
        "safe": {
            "target_func": "get_all_genres",
            "inject": 'cursor.execute("SELECT GenreId, Name FROM Genre ORDER BY Name")',
        },
    },
    # ── RQ1_3: IntegrityError 处理 ──
    "RQ1_3": {
        "vulnerable": {
            "target_func": "update_playlist_tracks",
            "inject": "cursor.execute(sql, params)  # 无 try/except",
            "ensure_absent": ["try:", "IntegrityError", "except"],
        },
        "safe": {
            "target_func": "update_playlist_tracks",
            "inject": '''
    try:
        cursor.execute(sql, params)
    except sqlite3.IntegrityError:
        pass  # 跳过重复记录
''',
        },
    },
    # ... 其他 criterion
}
```

### 对比挖掘算法(确定性,不需要 AI)

参考: **GumTree** (Falleri et al., 2014), **DECKARD** (Jiang et al., 2007)

```python
def contrastive_mine(vulnerable_codes: list[str], safe_codes: list[str],
                     min_vuln_rate: float = 0.6, max_safe_rate: float = 0.15,
                     ) -> dict:
    """从 VULNERABLE vs SAFE 代码中提取区分性特征。
    
    参考文献:
      - GumTree (Falleri et al., 2014): AST 树匹配与差分
      - DECKARD (Jiang et al., 2007): 子树向量化聚类
    """
    features = {}

    # 1. Token n-gram 对比
    vuln_tokens = [_tokenize(c) for c in vulnerable_codes]
    safe_tokens = [_tokenize(c) for c in safe_codes]

    for n in [1, 2, 3, 4]:  # unigram ~ 4-gram
        features[f'token_{n}gram'] = _find_discriminative_ngrams(
            vuln_tokens, safe_tokens, n, min_vuln_rate, max_safe_rate
        )

    # 2. AST 子树对比
    vuln_asts = [_parse_ast(c) for c in vulnerable_codes]
    safe_asts = [_parse_ast(c) for c in safe_codes]

    # 提取在 VULNERABLE 中频繁出现、SAFE 中罕见的 AST 子树
    features['ast_subtrees'] = _find_discriminative_ast_subtrees(
        vuln_asts, safe_asts, min_vuln_rate, max_safe_rate
    )

    # 3. 字符串字面量对比
    features['string_literals'] = _find_discriminative_strings(
        vulnerable_codes, safe_codes
    )

    return features
```

### 特征增强的 FS 生成 Prompt

将对比挖掘结果喂给 AI,降低其任务难度。参考: **Execution-Guided Synthesis** (Chen et al., 2019) 的执行→反馈→生成范式。

```
Current prompt (失败):
  "This code IS vulnerable. Write a check function."
  → AI 需要自己发现区分性特征 ← 太困难

Enhanced prompt (预期改善):
  "This code IS vulnerable. Here are the features that distinguish
   it from safe code:

   Token n-grams only in VULNERABLE:
     - 'f" ORDER BY {' (出现在 10/11 VULNERABLE, 0/39 SAFE)
     - 'sort_column' in ORDER BY (出现在 11/11 VULNERABLE, 2/39 SAFE)

   AST patterns only in VULNERABLE:
     - JoinedStr inside Call(func=Attribute(attr='execute')) (11/11 vs 0/39)

   String patterns only in SAFE:
     - 'allowed' dict with .get() (出现在 0/11 VULNERABLE, 35/39 SAFE)

   Based on these features, write a check function."
   → AI 只需组装特征 ← 难度大幅降低
```

### 预期效果

| 指标 | 当前 (方案 7) | 方案 8 保守 | 方案 8 乐观 | 增量来源 |
|------|:----------:|:---------:|:---------:|------|
| P | 0.146 | 0.45 | 0.70 | 对比挖掘提供区分性特征,AI 只需组装 |
| R | 0.390 | 0.55 | 0.75 | 确定性注入保证标签正确,干净对比对 |
| F1 | 0.213 | 0.50 | 0.72 | |

核心改进逻辑:
1. **确定性注入** → CW 的标签 100% 可靠(不再有 S005 式的 README/行为 不一致)
2. **对比挖掘** → 从干净对比对中提取真正区分性的 token/AST 特征
3. **特征增强 prompt** → AI 从"自己发现特征"降级为"组装已知特征"→ 成功率从 1/4 提升到接近 100%

### 为什么这个方案能突破当前瓶颈

CLAUDE.md 中记录的问题 1 (语法方法上限 P=1.0, R=0.05) 的根因是:
> "不存在语法层面的区分特征。TP 和 FP 代码在该 criterion 相关的代码中有着完全相同的 AST 结构和字符串模式。"

但这个问题有一个隐含假设:**TP 和 FP 代码在语法上没有差异**。这个假设在 CW 用 AI 生成代码时成立——因为 AI 倾向于生成结构相似、只在意图上不同的代码(例如都在函数内用 f-string,只是有些加了 if 验证有些没有)。

确定性注入打破了这个假设:**人为确保 vulnerable 代码有某种语法特征、safe 代码没有**(或反过来)。这样对比挖掘就能找到真正区分性的特征。

### 实施步骤与用时

```
Step 1: CW 确定性注入器
  - 新建 CW-generater/pattern_injector.py             2h
  - 为 4 个可测试 criterion 定义模板对                 2h
  - 重新生成 50 个学生 (确定性注入版)                   0.5h (API)
  - 沙箱验证: 行为标签与注入标签 100% 一致              0.5h

Step 2: 对比挖掘器
  - 新建 runtime/contrastive_miner.py                  3h
  - Token n-gram 对比提取                              1h
  - AST 子树对比提取                                   2h

Step 3: 特征增强 FS 生成
  - 修改 build_negative_fs_prompt() 接受特征列表        1h
  - 修改 generate_fs_for_clusters() 调用对比挖掘        0.5h
  - 端到端测试 + blind test                            2h

总计: ~12.5h
```
