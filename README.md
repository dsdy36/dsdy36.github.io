# TAFFIES 自动化评分系统 (FS_generater)

为 Python 编程作业自动生成 Feedback Signature（check function + feedback），用于检测学生代码中的错误模式和正确实现。

## 项目结构

```
FURP/
├── FS_generater-v1/            ← TAFFIES FS 生成管线
│   ├── taffies_fs_generator.py → 核心：聚类 + AI 生成 check function + FCC 迭代
│   ├── ai_pipeline.py          → DeepSeek API、数据收集
│   ├── ground_truth.py         → README 解析（Error/Mistake 分离）
│   ├── coverage.py             → FCC 覆盖检查（仅 regex）
│   ├── blind_test.py           → 盲测：P/R/F1 vs Ground Truth
│   ├── round5_refine.py        → check function 正反例迭代精炼（最多 5 轮）
│   ├── round4_refine.py        → regex 精炼（legacy）
│   ├── ast_diff_fs.py          → AST 差分规则合成（实验性）
│   ├── audit.py                → 最终审计（仅 regex）
│   ├── audit_check_fns.py      → check function 质量审计
│   ├── analyze_fp.py           → FP 分布分析
│   ├── plan_d.py               → 编排器（Phase 0/1 + 后处理）
│   ├── sandbox.py              → reference 验证（Flask test_client）
│   ├── config_generator.py     → 提交扫描
│   ├── comment_stripper.py     → 注释过滤
│   ├── test_fs_generator.py    → 测试驱动 FS 生成器
│   ├── runtime/                → 行为沙箱模块
│   │   ├── subprocess_executor.py → subprocess 隔离执行
│   │   ├── test_generator.py      → 行为测试模板（4 个 criterion 的手写 harness）
│   │   └── batch_runner.py        → 批量行为测试 + 行为聚类
│   ├── question/code/          → iMusic.py 模板（含 stub 函数）
│   ├── references/             → AI 生成的 reference 函数
│   └── output/q1_iMusic/       → fs_registry.json + 行为指纹 + 行为聚类
│
├── CW-generater/               ← 学生提交生成器（训练数据源）
│   ├── main_v5.py              → 入口: python main_v5.py -n 50
│   ├── pattern_matrix.py       → rubric → patterns × variants（硬编码 + AI 回退）
│   ├── generate_readme.py      → 生成 README（记录每个学生注入了哪些 pattern）
│   ├── convert_for_v1.py       → CW 格式 → V1 submission 格式
│   ├── inject_sql_patterns.py  → SQL 注入后处理
│   ├── fix_vague_labels.py     → AI 批量重写模糊 CW label
│   ├── check_readme_quality.py → README 质量检查
│   ├── check_remaining.py      → 残余模糊 label 检查
│   ├── test_phase0.py          → Phase 0 测试
│   └── question/code/          → 同 FS_generater-v1/question/code/
│
└── taffies-2026-main/          ← taffies-2026 参考实现（TypeScript, TanStack Start）
    └── src/lib/server/runtime/ → Docker 沙箱、AI harness 探索、workspace 生成
```

## 完整流程

```bash
# === CW 端：生成训练数据 ===
cd CW-generater
python main_v5.py -n 50                    # 1. 生成 50 个学生提交
python generate_readme.py submissions_imusic_v5 50  # 2. 生成 README
python convert_for_v1.py submissions_imusic_v5 ../FS_generater-v1/submission  # 3. 格式转换

# === FS 端：生成 Feedback Signature ===
cd ../FS_generater-v1
python taffies_fs_generator.py question submission q1_iMusic  # 4. 生成 FS
python round5_refine.py output/q1_iMusic/fs_registry_taffies.json submission  # 5. 可选：迭代精炼
python blind_test.py output/q1_iMusic/fs_registry_taffies.json submission   # 6. 盲测
```

### 管线内部流程

```
Phase 0:  AI 从 PDF 提取 rubric → rubric_cache.json（缓存，仅首次需要 API）
Phase 1:  AI 生成 reference 函数片段 → 沙箱验证

TAFFIES FS 生成:
  A. 聚类: 从 50 个 README 中读取每个学生注入了哪个具体 pattern variant
     → 相同 pattern label 的学生归为一组（如 "pandas" 组、"readlines" 组）
     → 注意：聚类依据是 label 文本相似度，不是代码相似度
  B. 生成: 每个 cluster → AI 看到全部学生的 criterion 相关函数体
     → 生成 1 条 check function（Type B: 检测 PRESENCE）
     → check function 签名: def check(code: str) -> tuple: return (bool, "evidence")
     → System prompt 禁止 Type A（return not, "not in"）
  C. FCC:  迭代补缺（最多 3 轮）
     → 优先补负向 FS（用 rubric bad_patterns 作为 label）
  D. 验证: 负向 FS 匹配 reference → 降权 weight=0.5
           正向 FS 匹配 template → 降权 weight=0.5

后处理（plan_d.py）:
  - Criteria Filter: 无 bad pattern 的 criterion → 删除该 criterion 所有负向 FS
  - Broad Filter: 匹配率 >40% 且 FPR >50% → 降权（GT-aware）
  - Variable Gen: 用 \w+ 替换非白名单变量名

输出: fs_registry.json（每条 FS 含 check_function + feedback + marks + evidence）
```

## 核心概念

- **FS (Feedback Signature)** = `check_function` (Python `def check(code) -> tuple`) + `feedback` (学生可读文本) + `marks` (分值) + `evidence` (匹配证据字符串)。check function 返回 `(True, "evidence")` 表示匹配成功，`(False, "")` 表示未匹配。
- **Type A vs Type B**: Type A 检测"缺失某好模式"（如 `return commit() not in code`）→ 精度低。Type B 检测"存在某坏模式"（如 `return 'import pandas' in code`）→ 精度高。System prompt 只允许 Type B。
- **FCC (Feedback Coverage Checker)**: 每个学生每个 criterion 至少 1 个 FS 匹配。用于迭代补充。
- **Ground Truth**: CW 生成的 README.md 精确记录每个学生注入了哪些 pattern variant。
- **14 个评分标准**: Task1: RQ1_1-RQ1_4, Task2: RQ2_1-RQ2_4, Task3: RQ3_1-RQ3_6。
- **Error vs Mistake**: Error pattern 一定是错的（f-string SQL 注入、pandas）。Mistake 可能出现在正确代码中（INSERT OR IGNORE、SELECT COUNT before INSERT）→ 生成负向 FS 但 prompt 告知 AI "只在缺少伴随检查时标记"。

## 运行环境

- Python: `C:\ProgramData\anaconda3\python.exe`
- DeepSeek API: `.env` 在 `FS_generater-v1/` 和 `CW-generater/`
- DeepSeek Model: `deepseek-chat`（默认），`deepseek-reasoner` 可选（需处理 system→user message 转换）
- DeepSeek 上下文: 64K tokens，代码用 50K chars 保守上限
- 依赖: `openai`, `PyPDF2`, `pyyaml`, `python-dotenv`, `flask`

## 当前盲测结果

2026-07-09 最新运行（132 FS: 110 positive + 22 negative, 50 students, 14 criteria）：

```
Overall:  P=0.15  R=0.62  F1=0.24
          TP=51    FP=300  FN=31

Per-criterion negative FS 表现：
  RQ1_4:  P=0.60  R=1.00  F1=0.75
  RQ1_3:  P=0.36  R=0.80  F1=0.50
  RQ2_3:  P=0.24  R=0.78  F1=0.37
  RQ3_5:  P=0.20  R=1.00  F1=0.33
  其余 8 个 criterion 的 F1 在 0.00~0.26 之间
  300 个 FP 中 88% 来自 9 条过于宽泛的 FS
```

## 当前问题

### 根本问题：语法方法的上限

14 个 criterion 中，只有 RQ1_3 和 RQ1_4 有可用的 FS。其他 criterion 的 check function 均存在大量 FP，根因是**语法层面不存在区分特征**。

以 RQ2_3 "ORDER BY 需要 whitelist 验证" 为例：

两个学生的代码核心部分完全一样（都用了 `f" ORDER BY {col}"`），区别在于一个前面有 whitelist 验证，一个没有。但验证逻辑有无数种写法：dict.get()、if/elif 链、set membership、函数封装——而"没做验证"共享的唯一语法特征恰好是 f-string，但这个特征在正确代码里也存在。

**错误是"没有阻止 SQL 注入"——这是行为属性，不是文本属性。** 13/14 个 criterion 都属于此类。

### 行为沙箱可靠但无法接入

runtime/ 下的行为沙箱（subprocess 执行学生代码 + mock sqlite3 + 注入恶意输入）可以完美区分 VULNERABLE/SAFE：200/200 次测试成功，零超时，分离清晰。但行为测试结果目前是外部批量脚本，无法封装成单条 `def check(code) -> tuple`。

方案 7 尝试让 AI 生成包含沙箱执行能力的 check function：4 个 criterion 中 3 个 FAILED——AI 写不好完整的 mock 层（漏 mock get_db_connection()、Path.mkdir() 等依赖）。

## 所有方案结果汇总

| 方案 | P | R | F1 | 核心问题 |
|------|:--:|:--:|:--:|------|
| 1. Plan D (regex) | 0.18 | 0.90 | 0.30 | Type A 过度匹配，447 FP |
| 2. TAFFIES check fn (语法) | 0.29 | 0.37 | 0.29 | Type B 改善但 AI 质量方差大 |
| 3. TAFFIES + FPR 过滤 | 1.00 | 0.05 | 0.10 | 仅 3 条 FS 达到 P=1.0 |
| 4. Round 5 迭代精炼 | 0.11 | 0.37 | 0.17 | AI 无法收紧而不失 TP |
| 5. AST Diff 规则合成 | — | — | — | 无区分性语法特征 |
| 6. 测试驱动 v1 (简陋沙箱) | 0.28 | 0.28 | 0.28 | mock 环境不准确 |
| 7. 测试驱动 v2 (Flask) | 0.29 | 0.19 | 0.23 | 沙箱不稳定，5 criterion 全败 |
| 8. 沙箱先行 (subprocess) | 0.15 | 0.39 | 0.21 | 行为数据无法自动转化为语法特征 |
| 当前 (多学生代码+行为辅助) | 0.15 | 0.62 | 0.24 | 精度问题未解决，300 FP |

## 可考虑的解决方向

### 方向 A：人手写行为测试模板 + AI 做模板匹配

- 核心思路：放弃让 AI 生成 check function，改为手写 14 个 criterion 的行为测试模板（mock + probe），AI 只负责判断"这个学生代码匹配哪个模板"
- 优点：行为测试已证明可靠（200/200），人手写 mock 是一次性投入
- 缺点：AI 角色退化，模板对代码变体的泛化能力未知

### 方向 B：确定性注入 + 对比挖掘

- 核心思路：CW 端确保 vulnerable/safe 代码只在关键 pattern 上有差异 → FS 端统计对比提取区分性特征 → AI 基于已知特征组装 check function
- 优点：打破了"TP/FP 语法无差异"的隐含假设
- 缺点：只解决 CW 端的区分问题，真实学生代码可能没有这种干净对比；需要实现 pattern_injector.py + contrastive_miner.py

### 方向 C：LLM-as-Judge 直接评分

- 核心思路：不再生成 check function，直接把学生代码发给 LLM，让它根据 rubric 判断是否正确
- 优点：LLM 可以理解语义，不需要语法特征
- 缺点：成本高（50 students × 14 criteria = 700 次 API 调用/轮），一致性不可控，无法离线部署

### 方向 D：代码嵌入 + 分类器

- 核心思路：将学生代码转为向量（CodeBERT/CodeT5），在行为标签上训练分类器
- 优点：可能捕获语法方法漏掉的模式
- 缺点：需要大量标注数据，50 个学生可能不够；可解释性差（无法生成学生可读的 feedback）

## 文献出处

| 文献 | 在本项目中的应用 | 文件 |
|------|------|------|
| TAFFIES (Pike, Lee & Towey, 2022) | FS = signature + feedback 配对；criterion-by-criterion 聚类；FCC 覆盖循环；Type A/B 区分 | taffies_fs_generator.py |
| OverCode (Glassman et al., 2015) | 学生代码变体可视化与聚类 | cluster_by_pattern() |
| GumTree (Falleri et al., 2014) | AST 级代码差分算法 | ast_diff_fs.py |
| Execution-Guided Synthesis (Chen et al., 2019) | 执行结果反馈指导代码生成 | runtime/test_generator.py |
| Test-Driven Repair (Long & Rinard, 2016) | 测试驱动代码修复 | round5_refine.py |
| DECKARD (Jiang et al., 2007) | AST 子树相似度聚类 | coverage.py |
| Keuning et al. (2018) | 自动反馈生成系统文献回顾，确认"语法特征区分性不够"是普遍问题 | CLAUDE.md |
| Messer et al. (2024) | LLM-based feedback 最新综述，确认 LLM 在理解代码意图/行为方面有根本困难 | CLAUDE.md |
