# LLM-DA-for-public-github

一个面向时间知识图谱（Temporal Knowledge Graph, TKG）的规则推理实验仓库。仓库把几类能力串成了一条完整实验链路：

- 随机游走采样与原始规则抽取
- 基于 LLM 的规则生成、清洗与迭代改写
- 规则置信度估计与规则排序
- 规则推理、候选答案生成与图模型分数融合
- 细粒度 trace、关系级诊断、流行度偏置分析
- 利用 LLM 对规则置信度进行评估与调整

当前代码里的自动迭代入口是 [`run_vlrg_loop.py`](./run_vlrg_loop.py)。它会在 valid 集上做多轮“评估 -> 误差分析 -> 生成证据 -> 让大模型评估规则置信度并给出调整建议 -> 应用调整 -> 重新推理”，并按 valid MRR 选出最佳轮次，再可选地在 test 集上复测。

## 1. 整体流程

```text
rule_sampler.py
  -> results/<bat>/sampled_path/<dataset>/

Iteration_reasoning.py
  -> results/<bat>/gen_rules_iteration/<dataset>/final_summary/rules.txt

rank_rule.py
  -> results/<bat>/ranked_rules/<dataset>/confidence.json
  -> results/<bat>/ranked_rules/<dataset>/confidence_concrete.json

reasoning.py
  -> cands_*.json
  -> trace_*.jsonl

evaluate.py
  -> metrics_*.json / metrics_*.csv

analyze_eval_breakdown.py
diagnose_relations.py
diagnose_popularity_bias.py
build_llm_evidence.py
  -> breakdown_*.json
  -> rel_diag_*.json
  -> popbias_*.json
  -> evidence_for_llm_*.json

llm_rule_governance.py
  -> patch_round_*.json

apply_rule_patch.py
  -> confidence_round_{k+1}.json
  -> confidence_concrete_round_{k+1}.json
```

如果你只关心“已有规则后怎么跑大模型规则置信度评估流程”，重点看：

1. 先准备 `sampled_path` 产物和 `gen_rules_iteration/.../final_summary/rules.txt`
2. 准备图模型导出的 `test.npy/score.npy` 与 `test_valid.npy/score_valid.npy`
3. 直接运行 `run_vlrg_loop.py`

## 2. 目录说明

| 路径 | 作用 |
| --- | --- |
| `datasets/` | 数据集目录，仓库内已包含 `icews14`、`icews0515`、`icews18`、`GDELT` |
| `prompt/` | LLM 规则生成与迭代时使用的 prompt 模板 |
| `llms/` | LLM 适配层。默认 OpenRouter / OpenAI 风格接口，也保留本地 HF 模型适配器 |
| `Config/constant.json` | 关系正则、规则清洗相关常量 |
| `rule_sampler.py` | 随机游走采样路径并生成 `sampled_path` 目录 |
| `Iteration_reasoning.py` | 使用 prompt + 路径样本进行 LLM 规则生成、清洗、迭代 |
| `rank_rule.py` | 计算规则置信度并生成 `confidence.json` / `confidence_concrete.json` |
| `reasoning.py` | 将规则应用到 valid/test 查询，输出候选实体和 trace |
| `evaluate.py` | 评估候选结果，支持与图模型分数融合 |
| `analyze_eval_breakdown.py` | 统计 NoCand / NoHit / HitTop1 / HitNotTop1 等 breakdown |
| `diagnose_relations.py` | 关系级错误分析 |
| `diagnose_popularity_bias.py` | 流行度偏置分析 |
| `build_llm_evidence.py` | 将 trace 与误差分析结果压缩成供 LLM 评估规则置信度的证据包 |
| `llm_rule_governance.py` | 让 LLM 根据证据输出规则置信度调整建议 |
| `apply_rule_patch.py` | 对 `confidence*.json` 应用置信度调整，得到下一轮规则集 |
| `run_vlrg_loop.py` | 自动执行多轮规则置信度评估与重新推理 |
| `semantics_info_builder.py` | 离线构建 relation profile，供 prompt 语义增强使用 |
| `example_pool_builder.py` | 离线构建动态 few-shot 示例池 |

## 3. 环境准备

### Python

仓库本地环境为 Python 3.11，推荐使用 Python 3.10+。

### 依赖

仓库目前没有提供 `requirements.txt`，建议至少安装下面这些包：

```bash
pip install numpy scipy pandas torch joblib tqdm networkx openai python-dotenv tiktoken
pip install openpyxl sentence-transformers scikit-learn transformers
```

说明：

- `openpyxl` 只在导出 Excel 或示例池统计表时需要。
- `sentence-transformers` 和 `scikit-learn` 主要用于 `rule_sampler.py` 的关系相似度矩阵构建。
- `transformers` 主要用于本地 HF 模型适配或相关工具脚本。

### 环境变量

如果你使用 OpenRouter 驱动的模型（当前 `llm_rule_governance.py` 默认就是这种模式），在仓库根目录创建 `.env`：

```env
OPENROUTER_API_KEY=your_api_key_here
```

`llms/chatgpt.py` 会自动加载 `.env`。

### 一个重要约定

很多脚本都带 `--bat_file_name` 和 `--results_root_path`：

- `bat_file_name` 可以理解为实验名或实验分组名
- 所有中间结果都会写到 `results/<bat_file_name>/...`
- 某些脚本的参数解析对 `--results_root_path` 比较敏感，建议把它放在命令最后

## 4. 数据与输入要求

### 4.1 数据集格式

每个数据集目录至少需要下面这些文件：

```text
datasets/<dataset>/
  entities.txt
  relations.txt
  train.txt
  valid.txt
  test.txt
  entity2id.json
  relation2id.json
  ts2id.json
```

其中三元/四元组文件格式为：

```text
subject<TAB>relation<TAB>object<TAB>timestamp
```

### 4.2 图模型基线文件

如果你要运行 `evaluate.py`、`reasoning.py --trace_rules Yes` 或 `run_vlrg_loop.py` 的默认融合流程，还需要准备图模型导出的 `.npy` 文件：

```text
datasets/<dataset>/<graph_reasoning_type>/
  test.npy
  score.npy
  test_valid.npy
  score_valid.npy
```

当前代码中显式支持的图模型类型主要是：

- `TiRGN`
- `REGCN`
- `LogGL`

如果这些文件缺失，和图模型融合相关的评估/trace 会直接报错。

### 4.3 数据集兼容性说明

仓库里虽然带了 `GDELT` 目录，但 `Config/constant.json` 里的关系正则目前只配置了：

- `icews14`
- `icews0515`
- `icews18`

也就是说，涉及规则字符串解析、规则清洗、LLM 规则回写的流程默认是按 ICEWS 风格配置的。若要在 `GDELT` 上完整跑通 LLM 规则链路，需要先补充对应的 relation regex。

## 5. 推荐运行路径

### 5.1 先采样路径与原始规则

`Iteration_reasoning.py` 通常需要同时用到 train 与 valid 两套 `sampled_path` 产物，建议至少跑两次：

```bash
python rule_sampler.py -d icews14 --version train --max_path_len 3 -n 100 -p 8 --bat_file_name exp01 --results_root_path results
python rule_sampler.py -d icews14 --version valid --max_path_len 3 -n 100 -p 8 --bat_file_name exp01 --results_root_path results
```

这一步会生成类似下面的内容：

```text
results/exp01/sampled_path/icews14/
results/exp01/sampled_path/icews14_valid/
```

其中常用文件包括：

- `closed_rel_paths.jsonl`
- `rules_name.json`
- `rules_var.json`
- `confidence_concrete.json`
- `matrix.npy`

### 5.2 使用 LLM 生成/迭代规则

最常见的规则生成入口是：

```bash
python Iteration_reasoning.py --dataset icews14 --model_name gpt-5-nano --bgkg valid --num_iter 2 -n 8 --eval_workers 8 --base_seed 1 --bat_file_name exp01 --results_root_path results
```

默认情况下，这一步会把结果写到：

```text
results/exp01/gen_rules_iteration/icews14/
```

其中 `final_summary/rules.txt` 是后续 `rank_rule.py` 和 `run_vlrg_loop.py` 默认读取的规则输入。

可选增强：

- `--use_semantic_profile yes`：启用关系语义 profile 注入
- `--use_dynamic_examples --example_pool_path <path>`：启用动态 few-shot 示例池
- `--dry_run`：只走流程，不真正调用 LLM

### 5.3 一键运行 VLRG 规则置信度评估流程

在下面三个前置条件都满足后：

1. `results/exp01/sampled_path/<dataset>/...` 已存在
2. `results/exp01/gen_rules_iteration/<dataset>/final_summary/rules.txt` 已存在
3. `datasets/<dataset>/<graph_reasoning_type>/score*.npy` 已存在

可以直接运行：

```bash
python run_vlrg_loop.py -d icews14 --K 3 --graph_reasoning_type TiRGN --rule_weight 0.9 --llm_model gpt-5-nano --trace_rules Yes --incremental_reasoning Yes --final_test Yes --bat_file_name exp01 --results_root_path results
```

常用参数：

- `--K`：规则置信度评估的迭代轮数
- `--llm_dry_run`：跳过真实 LLM 评估，生成空调整文件便于联调
- `--trace_rules Yes|No`：是否保留规则归因 trace
- `--incremental_reasoning Yes|No`：是否只对受置信度调整影响的 query 增量重算
- `--query_dir both|forward`：是否只评估正向关系
- `--test_all_rounds Yes`：把每一轮都在 test 上复测
- `--final_test Yes`：对 valid 最优轮次做最终 test

## 6. 已有规则时的最短路径

如果你已经有规则文件，不想从头跑 `Iteration_reasoning.py`，最少需要准备：

```text
results/<bat>/sampled_path/<dataset>/original/rules_var.json
results/<bat>/gen_rules_iteration/<dataset>/final_summary/rules.txt
datasets/<dataset>/<graph_reasoning_type>/test.npy
datasets/<dataset>/<graph_reasoning_type>/score.npy
datasets/<dataset>/<graph_reasoning_type>/test_valid.npy
datasets/<dataset>/<graph_reasoning_type>/score_valid.npy
```

然后直接执行上一节的 `run_vlrg_loop.py` 即可。

如果你不想用 LLM 生成的规则，而是只想对随机游走采样得到的规则排序，可以手工运行：

```bash
python rank_rule.py -d icews14 --random_walk_rules yes --bgkg train --bat_file_name exp01 --results_root_path results
```

## 7. 手工分步运行

如果你想调试单个阶段，而不是一次性跑完整规则置信度评估流程，可以按下面的顺序执行。

### 7.1 规则打分

```bash
python rank_rule.py -d icews14 --bgkg train --bat_file_name exp01 --results_root_path results
```

输出：

- `results/exp01/ranked_rules/icews14/confidence.json`
- `results/exp01/ranked_rules/icews14/confidence_concrete.json`

### 7.2 推理并写出候选/trace

```bash
python reasoning.py -d icews14 --test_data valid -r confidence.json --bgkg all --graph_reasoning_type TiRGN --rule_weight 0.9 --trace_rules Yes --candidates_file_name cands_debug.json --trace_file_name trace_debug.jsonl --bat_file_name exp01 --results_root_path results
```

### 7.3 评估

```bash
python evaluate.py -d icews14 --test_data valid --candidates_file_name cands_debug.json --graph_reasoning_type TiRGN --rule_weight 0.9 --bat_file_name exp01 --results_root_path results
```

### 7.4 误差分析与证据包

```bash
python analyze_eval_breakdown.py -d icews14 --test_data valid --candidates_file cands_debug.json --graph_reasoning_type TiRGN --rule_weight 0.9 --trace_file trace_debug.jsonl --bat_file_name exp01 --results_root_path results

python diagnose_relations.py -d icews14 --test_data valid --candidates_file cands_debug.json --graph_reasoning_type TiRGN --rule_weight 0.9 --trace_file trace_debug.jsonl --bat_file_name exp01 --results_root_path results

python diagnose_popularity_bias.py -d icews14 --test_data valid --candidates_file cands_debug.json --graph_reasoning_type TiRGN --rule_weight 0.9 --trace_file trace_debug.jsonl --bat_file_name exp01 --results_root_path results

python build_llm_evidence.py -d icews14 --round 0 --test_data valid --graph_reasoning_type TiRGN --rule_weight 0.9 --candidates_file cands_debug.json --trace_jsonl trace_debug.jsonl --confidence_concrete_file confidence_concrete.json --bat_file_name exp01 --results_root_path results
```

### 7.5 让大模型评估规则置信度并应用调整

```bash
python llm_rule_governance.py --evidence_file results/exp01/ranked_rules/icews14/evidence_for_llm_<run_id>_TiRGN_w0.9.json --model_name gpt-5-nano

python apply_rule_patch.py --confidence_in results/exp01/ranked_rules/icews14/confidence_round_000.json --confidence_concrete_in results/exp01/ranked_rules/icews14/confidence_concrete_round_000.json --patch_file results/exp01/ranked_rules/icews14/patch_round_001.json --out_dir results/exp01/ranked_rules/icews14
```

## 8. 关键输出文件

`run_vlrg_loop.py` 的主要结果都在：

```text
results/<bat_file_name>/ranked_rules/<dataset>/
```

常见文件含义如下：

| 文件 | 说明 |
| --- | --- |
| `cfg_<cfg_id>.json` | 当前实验配置的哈希快照 |
| `confidence_round_000.json` | 第 0 轮规则置信度 |
| `confidence_concrete_round_000.json` | 第 0 轮规则置信度 + 可读规则字符串 |
| `cands_<run_id>.json` | 每个 query 的候选实体与分数 |
| `trace_<run_id>_<graph>_w*.jsonl` | 每个 query 的 top1/GT 规则归因信息 |
| `metrics_<run_id>_<graph>_w*.json` | Hits@1/3/10、MRR 等指标 |
| `breakdown_<run_id>_<graph>_w*.json` | NoCand / NoHit / HitTop1 / HitNotTop1 breakdown |
| `rel_diag_<run_id>_<graph>_w*.json` | 关系级诊断结果 |
| `popbias_<run_id>_<graph>_w*.json` | 流行度偏置诊断结果 |
| `evidence_for_llm_<run_id>_<graph>_w*.json` | 提供给 LLM 做规则置信度评估的证据包 |
| `patch_round_00k.json` | LLM 生成的规则置信度调整建议 |
| `loop_summary.csv` | 每一轮 valid 结果与耗时汇总 |
| `best_round.txt` | valid 上的最佳轮次 |
| `test_summary.csv` | 所有轮次 test 结果（可选） |
| `final_test_metrics.json` | 最佳 valid 轮次的最终 test 指标（可选） |

## 9. 可选离线工具

### 9.1 关系语义 profile

用于给 `Iteration_reasoning.py` 的 common 阶段 prompt 注入关系统计语义：

```bash
python semantics_info_builder.py --dataset icews14 --split train --bat_file_name exp01 --results_root_path results
```

默认输出：

```text
results/exp01/semantics/icews14/relation_profile.json
```

注意：默认依赖 `datasets/<dataset>/facts_matched_rows.txt`。仓库里 `icews14` 自带该文件，但其他数据集不一定有。

### 9.2 动态 few-shot 示例池

用于给 `Iteration_reasoning.py --use_dynamic_examples` 提供本地示例库：

```bash
python example_pool_builder.py --dataset icews14 --bat_file_name exp01 --results_root_path results
```

默认输出：

```text
results/exp01/example_pool/icews14/example_pool.jsonl
results/exp01/example_pool/icews14/example_pool_stats.xlsx
```

注意：它默认会读取：

- `results/<bat>/sampled_path/<dataset>/confidence_concrete.json`
- `results/<bat>/gen_rules_iteration_for_example_pool/<dataset>/evaluation/train/0_confidence_concrete.json`

也就是说，若要使用默认路径，需要先额外准备 `gen_rules_iteration_for_example_pool` 目录。

## 10. 常见问题

### 10.1 `run_vlrg_loop.py` 一开始就报 `rules.txt` 找不到

`rank_rule.py` 默认会去读：

```text
results/<bat>/gen_rules_iteration/<dataset>/final_summary/rules.txt
```

先运行 `Iteration_reasoning.py`，或者改成手工执行 `rank_rule.py --random_walk_rules yes`。

### 10.2 `evaluate.py` / `reasoning.py` 报缺少 `score.npy`

这通常表示图模型基线尚未导出。请先把外部模型导出的 `test.npy` / `score.npy` / `test_valid.npy` / `score_valid.npy` 放到：

```text
datasets/<dataset>/<graph_reasoning_type>/
```

### 10.3 `OPENROUTER_API_KEY is not set`

在仓库根目录配置 `.env`，至少包含：

```env
OPENROUTER_API_KEY=...
```

### 10.4 写结果时报 `PermissionError`

仓库里很多 CSV / JSON / Excel 文件会被反复覆盖。如果你正用 Excel 打开输出文件，先关闭再重跑。

### 10.5 Windows 路径太长

仓库已经内置了 `utils_windows_long_path.py` 做兼容；在 Windows 上尽量继续使用仓库自带脚本，不要手工改成普通 `open()`。

## 11. 一个最小复现实验建议

如果你第一次接触这个仓库，建议按下面顺序最小化验证：

1. 运行两次 `rule_sampler.py`，先得到 `sampled_path/<dataset>` 和 `sampled_path/<dataset>_valid`
2. 准备好图模型导出的 `.npy`
3. 用 `Iteration_reasoning.py --dry_run` 验证规则生成链路
4. 用 `run_vlrg_loop.py --llm_dry_run --K 1` 验证自动评估流程的文件依赖与路径组织
5. 最后再去掉 dry-run，跑真实 LLM 调用

这样最容易快速定位是“数据缺失”“图模型缺失”还是“LLM 配置缺失”。
