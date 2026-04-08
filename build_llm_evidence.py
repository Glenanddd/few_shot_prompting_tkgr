"""
VLRG evidence_for_llm_*.json 指标定义（带例子）
================================================

本脚本把：
  - 推理归因日志 `trace_*.jsonl`（来自 `reasoning.py --trace_rules Yes`）
  - 评测 breakdown `breakdown_*.json`（来自 `analyze_eval_breakdown.py`）
  - 关系级诊断 `rel_diag_*.json`（来自 `diagnose_relations.py`）
  - 热门尾偏置 `popbias_*.json`（来自 `diagnose_popularity_bias.py`，可选）
  - 规则元数据 `confidence_concrete_*.json`（rule_id -> abstract_rule）
压缩成一个“证据包”，供 LLM 在 valid 上做规则治理（delete / downweight / promote）。

注意：本文档是“证据字段的统计口径”，不是算法论文。所有统计都只基于 trace/diagnostics 的可复现实验输出。

----------------------------------------------------------------------
1) trace_*.jsonl：一行 = 一个 query 的归因
----------------------------------------------------------------------
trace 由 reasoning.py 写入，每行 JSON 至少包含：
  - qid: int                     # query index（与 candidates.json 的 key 对齐）
  - query: [h, r, gt, time]      # 4 元组（gt 即 gt_tail）
  - relation: int                # r
  - gt_tail: int                 # ground truth tail id
  - pred_top1: int               # 融合后 top1 的 tail id
  - gt_rank_in_cands: int|null   # gt_tail 在 candidates 排名（1=最好）；不在候选则 null
  - cand_size: int               # 候选数量（len(candidates[qid])）
  - top1_rules: list[dict]       # 对 pred_top1 的规则归因（top-K）
  - ans_rules:  list[dict]       # 对 gt_tail 的规则归因（top-K）

top1_rules/ans_rules 的元素结构：
  {"rule_id": str, "delta": float, "abstract_rule": str, "confidence": float}

其中 delta 的定义来自 reasoning.py:_topk_rule_deltas_for_entity（按当前实现）：
  - score_type == 'noisy-or':
      对“规则侧的 noisy-or 聚合”做边际贡献分解（再乘 rule_weight）：
        delta = rule_weight * (S(all) - S(without this rule))
      代码实现等价形式：
        delta = rule_weight * s_rule * Π_{other}(1 - s_other)
  - score_type == 'sum':
        delta = rule_weight * sum(rule_scores)
  - score_type == 'mean'/'min'/'max' 等：
        近似：delta = rule_weight * agg(rule_scores)
    （近似意味着它不严格等价于最终融合分数的“真正边际贡献”，但可稳定比较规则贡献大小）

特殊来源（当某实体得分完全来自图推理 / 或 trace 无法计算规则贡献）：
  - rule_id == "GRAPH":          该实体主要来自 graph baseline（baseline 存在）
  - rule_id == "GRAPH_MISSING":  baseline 缺失（无法做 graph-only 对比/归因）
  - rule_id == "TRACE_ERROR":    trace 计算异常（仍写一行占位，避免 jsonl 断裂）

例子（概念示意）：
  某行 trace:
    {
      "qid": 7,
      "cand_size": 5,
      "gt_tail": 42,
      "pred_top1": 13,
      "gt_rank_in_cands": 3,
      "top1_rules": [{"rule_id":"a1b2c3", "delta":0.12, ...}, {"rule_id":"GRAPH", "delta":0.01, ...}],
      "ans_rules":  [{"rule_id":"d4e5f6", "delta":0.08, ...}]
    }
  含义：
    - GT 在候选里但不是 top1（rank=3 => HitNotTop1）
    - pred_top1=13 错；top1_rules 里 delta 最大的规则更可能是“推动错误 top1”的来源

----------------------------------------------------------------------
2) summary：全局 breakdown 指标（来自 analyze_eval_breakdown.py）
----------------------------------------------------------------------
N:
  - trace 行数（= query 数）

NoCand（比例）:
  - cand_size == 0 的 query / N

NoHit（比例）:
  - cand_size > 0 且 gt_rank_in_cands == null 的 query / N
  - 含义：系统给出了候选，但正确答案不在候选里（覆盖失败）

HitTop1（比例）:
  - cand_size > 0 且 pred_top1 == gt_tail 的 query / N

HitNotTop1（比例）:
  - cand_size > 0 且 pred_top1 != gt_tail 且 gt_rank_in_cands != null 的 query / N
  - 含义：候选覆盖了 GT，但排序没有把 GT 排到第一（排序失败）

HitNotTop1_rank_stats:
  - 仅对 HitNotTop1 子集收集 gt_rank_in_cands，输出其分布统计：
    {count,min,p50,p90,p95,p99,max,mean}

harm_rate / improve_rate / same_rate（比例；若 baseline 文件存在才计算，否则为 0）:
  - 对每条 query 做一次 “graph-only top1” vs “fused top1” 的对比（仅比较 top1 对错）：
    * harm:    graph_top1 == gt 但 fused_top1 != gt
    * improve: graph_top1 != gt 但 fused_top1 == gt
    * same:    其它情况（两者都对 / 两者都错 / baseline 缺失等）
  - 各 rate = count / N

例子：
  N=1000, NoCand=0.10 => 100 条 query 完全没有候选
  HitNotTop1_rank_stats.p95=20 => 在“候选覆盖但非 top1”的 query 里，95% 的 GT 排名 <= 20

----------------------------------------------------------------------
3) worst_relations：关系级诊断（来自 diagnose_relations.py + 可选 popbias）
----------------------------------------------------------------------
rel_diag_*.json 会按 relation 聚合 trace，并输出（这里挑关键字段）：
  - n (n_test):
      该 relation 的 query 数
  - NoHit（NoHit_rate_test）:
      gt_rank_in_cands == null 的数量 / n
      注意：这里的 NoHit 统计口径包含 cand_size==0（因为 gt_rank_in_cands 也为 null）
  - CondTop1_given_Hit:
      (pred_top1 == gt_tail 的数量) / Hit_count
      其中 Hit_count = (gt_rank_in_cands != null 的数量)
      解释：在“GT 被候选覆盖”的前提下，top1 命中的条件概率
  - WrongTop1_rate:
      (pred_top1 != gt_tail 的数量) / n
      注意：它也会包含 NoHit/NoCand，因为这两类一定 pred!=gt
  - Cand_p95:
      cand_size 的 95 分位（候选规模的上界感知）
  - AnsRankNotTop1_p95:
      在 pred!=gt 且 gt_rank_in_cands!=null 的子集里，gt_rank_in_cands 的 95 分位
  - TopWrongTop1Entities:
      pred_top1 错误时出现最多的实体 top10（eid,cnt）

popbias_*.json（可选）会给每个 relation 额外附加：
  - WrongTop1_top@10:
      该 relation 下最常见的错误 top1 实体 top10
  - TrainTail_top@{pop_top}:
      训练集中该 relation 最常见的 tail top{pop_top}
  - overlap_num:
      | top_wrong_eids ∩ top_train_eids |
  - overlapped_wrong_avg_train_rank:
      重叠实体在“训练 tail 热度全量排序”中的平均名次（1=最热）

例子：
  某 relation: n=200, NoHit=0.25 => 50 条 query 的 GT 不在候选里（或候选为空）
  CondTop1_given_Hit=0.40 => 在“GT 在候选里”的 150 条中，有 60 条 top1 命中

----------------------------------------------------------------------
4) harmful_rules：规则级诊断（本脚本从 trace 派生）
----------------------------------------------------------------------
我们用 trace 的 top1_rules / ans_rules 归因列表构建规则统计（跳过 rule_id=GRAPH/GRAPH_MISSING/TRACE_ERROR）：

trigger_top1:
  - 计数：该 rule_id 出现在 top1_rules(top-K) 的次数
  - 注意：不是“作为 top1_rules 第 1 名”的次数；进入 top-K 列表就算一次触发

harm_rate:
  - = trigger_top1_wrong / trigger_top1_total
  - trigger_top1_wrong：在 pred_top1 != gt_tail 的 query 上，该规则仍出现在 top1_rules 的次数
  - 值越大，说明该规则更常伴随错误 top1（优先 delete/downweight）

help_top1:
  - 计数：该 rule_id 出现在 ans_rules(top-K) 且该 query pred_top1 == gt_tail 的次数
  - 作为“规则可能帮助正确答案”的粗证据（不是因果证明）

avg_gt_rank_when_hit:
  - 在该 rule_id 出现在 ans_rules 且 gt_rank_in_cands != null 的这些 query 上，
    取 gt_rank_in_cands 的平均值（越小越好）

example_relations:
  - 该规则最常出现的 relation_id top10（帮助定位影响范围）

例子：
  trigger_top1=50, harm_rate=0.80 => 该规则进入 pred_top1 归因 top-K 共 50 次，其中 40 次最终 top1 是错的

----------------------------------------------------------------------
5) cases：从 worst_relations 抽样的具体样例（用于 LLM 读例子）
----------------------------------------------------------------------
从 worst_relations 选出的 rel_id 列表中，每个 relation 默认抽 cases_per_relation=3 条：
  - pass1：优先抽“错误样本”（pred!=gt 或 gt 不在候选 或 cand_size==0）
  - pass2：如果该 relation 不够 3 条，再补抽正常样本

每条 case 会携带 trace 的 top1_rules/ans_rules，LLM 可据此查看“哪些规则贡献大/是否偏向热门尾”等。
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grapher import Grapher
from utils_windows_long_path import maybe_windows_long_path, safe_open
from vlrg_utils import (
    candidates_run_id,
    configure_stdout_utf8,
    fmt_float_for_name,
    get_ranked_rules_dir,
    iter_jsonl,
    load_json,
    percentile_from_sorted,
    round_tag,
    safe_dump_json,
    topk_counter,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _default_file_names(graph_reasoning_type: str, rule_weight: float, candidates_file: str):
    run_id = candidates_run_id(candidates_file)
    w_str = fmt_float_for_name(rule_weight, decimals=2)
    return {
        "trace_jsonl": f"trace_{run_id}_{graph_reasoning_type}_w{w_str}.jsonl",
        "breakdown_json": f"breakdown_{run_id}_{graph_reasoning_type}_w{w_str}.json",
        "rel_diag_json": f"rel_diag_{run_id}_{graph_reasoning_type}_w{w_str}.json",
        "popbias_json": f"popbias_{run_id}_{graph_reasoning_type}_w{w_str}.json",
    }


def _load_confidence_concrete_rule_map(path: str) -> Dict[str, str]:
    payload = load_json(path, default={}) or {}
    mp: Dict[str, str] = {}
    for _, rules in payload.items():
        for r in rules or []:
            rid = r.get("rule_id")
            if not rid:
                continue
            mp[str(rid)] = r.get("abstract_rule", "") or ""
    return mp


def _load_candidates_dict(path: str) -> Dict[int, Dict[int, float]]:
    with safe_open(maybe_windows_long_path(path), "r", encoding="utf-8") as f:
        raw = json.load(f) or {}
    out: Dict[int, Dict[int, float]] = {}
    if not isinstance(raw, dict):
        return out
    for k, vv in raw.items():
        try:
            qid = int(k)
        except Exception:
            continue
        if not isinstance(vv, dict):
            out[qid] = {}
            continue
        m: Dict[int, float] = {}
        for ek, ev in vv.items():
            try:
                eid = int(ek)
                score = float(ev)
            except Exception:
                continue
            m[eid] = float(score)
        out[qid] = m
    return out


def _build_other_answers_map(split_np: np.ndarray) -> Dict[Tuple[int, int, int], List[int]]:
    mp: Dict[Tuple[int, int, int], List[int]] = {}
    for row in split_np:
        s = int(row[0])
        r = int(row[1])
        o = int(row[2])
        t = int(row[3])
        mp.setdefault((s, r, t), []).append(o)
    return mp


def _load_graph_baseline(dataset: str, dataset_dir: str, graph_reasoning_type: str, split: str):
    split_s = str(split or "test").lower()
    base_dir = maybe_windows_long_path(os.path.join(dataset_dir, graph_reasoning_type))
    if split_s == "valid":
        test_path = maybe_windows_long_path(os.path.join(base_dir, "test_valid.npy"))
        score_path = maybe_windows_long_path(os.path.join(base_dir, "score_valid.npy"))
        if not os.path.exists(score_path):
            typo = maybe_windows_long_path(os.path.join(base_dir, "score_vlid.npy"))
            if os.path.exists(typo):
                score_path = typo
    else:
        test_path = maybe_windows_long_path(os.path.join(base_dir, "test.npy"))
        score_path = maybe_windows_long_path(os.path.join(base_dir, "score.npy"))

    if not os.path.exists(test_path) or not os.path.exists(score_path):
        return None

    test_numpy = np.load(test_path)
    if str(dataset).lower() == "icews18":
        test_numpy[:, 3] = (test_numpy[:, 3] / 24).astype(int)
    score_numpy = np.load(score_path, mmap_mode="r")

    test_index: Dict[Tuple[int, int, int, int], int] = {}
    for i, row in enumerate(test_numpy):
        key = tuple(int(x) for x in row)
        if key not in test_index:
            test_index[key] = int(i)
    return score_numpy, test_index


def _argmax_with_exclude(scores_row: np.ndarray, exclude: List[int]) -> int:
    if not exclude:
        return int(np.argmax(scores_row))
    idx = int(np.argmax(scores_row))
    if idx not in exclude:
        return idx
    tmp = np.array(scores_row, dtype=np.float32, copy=True)
    for x in exclude:
        if 0 <= int(x) < tmp.shape[0]:
            tmp[int(x)] = -np.inf
    return int(np.argmax(tmp))


def _compute_breakdown_and_rel_diag_no_trace(
    *,
    dataset: str,
    split: str,
    graph_reasoning_type: str,
    rule_weight: float,
    candidates: Dict[int, Dict[int, float]],
    split_np: np.ndarray,
    score_numpy: Optional[np.ndarray],
    test_index: Optional[Dict[Tuple[int, int, int, int], int]],
    other_answers_map: Dict[Tuple[int, int, int], List[int]],
    grapher: Grapher,
    min_n: int = 50,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build breakdown(relations+summary) without trace:
    - pred_top1 is computed with the same fusion as reasoning.py trace:
        fused(e) = (1-w)*graph(e) + w*rule(e), rule(e)=0 for non-candidates
    - NoHit_rate_test in relation diag counts gt not in candidates (includes NoCand), consistent with diagnose_relations.py
    """

    n = int(len(split_np))
    no_cand = no_hit = hit_top1 = hit_not_top1 = 0
    hit_not_top1_ranks: List[int] = []
    harm = improve = same = 0

    # per relation accumulators (similar to diagnose_relations.py)
    n_by_rel = Counter()
    nohit_by_rel = Counter()  # gt not in candidates (including NoCand)
    hit_by_rel = Counter()
    top1_hit_by_rel = Counter()
    wrong_top1_by_rel = Counter()
    cand_sizes_by_rel: Dict[int, List[int]] = defaultdict(list)
    ans_rank_not_top1_by_rel: Dict[int, List[int]] = defaultdict(list)
    wrong_top1_entities_by_rel: Dict[int, Counter] = defaultdict(Counter)

    w = float(rule_weight)
    iterator = enumerate(split_np)
    if tqdm is not None:
        iterator = tqdm(iterator, total=n, desc="No-trace stats", unit="q", mininterval=0.5, file=sys.stdout)
    for qid, row in iterator:
        qid_i = int(qid)
        s = int(row[0])
        r = int(row[1])
        gt = int(row[2])
        ts = int(row[3])
        key = (s, r, gt, ts)

        cand_scores = candidates.get(qid_i, {}) or {}
        cand_size = int(len(cand_scores))

        # gt_rank_in_cands (rule candidates only)
        gt_rank_in_cands = None
        if gt in cand_scores:
            all_confs = sorted([float(x) for x in cand_scores.values()], reverse=True)
            try:
                gt_rank_in_cands = int(all_confs.index(float(cand_scores[gt])) + 1)
            except Exception:
                gt_rank_in_cands = None

        pred_top1 = -1
        graph_top1 = -1
        graph_found = False
        if score_numpy is not None and test_index is not None:
            pos = test_index.get(tuple(int(x) for x in key))
            if pos is not None:
                graph_found = True
                scores_row = score_numpy[int(pos)]
                objs = other_answers_map.get((s, r, ts), []) or []
                exclude = [int(o) for o in objs if int(o) != gt]
                graph_top1 = _argmax_with_exclude(scores_row, exclude)

                # candidate best under fused score
                best_cand = -1
                best_cand_score = -float("inf")
                if cand_scores:
                    for eid, rs in cand_scores.items():
                        if eid in exclude:
                            continue
                        try:
                            gs = float(scores_row[int(eid)])
                        except Exception:
                            gs = 0.0
                        fused = (1.0 - w) * gs + w * float(rs)
                        if fused > best_cand_score:
                            best_cand_score = fused
                            best_cand = int(eid)

                # baseline best under (1-w)*graph
                gs_best = float(scores_row[int(graph_top1)]) if graph_top1 >= 0 else -float("inf")
                best_graph_score = (1.0 - w) * gs_best if w < 1.0 else 0.0

                if best_cand >= 0 and best_cand_score > best_graph_score:
                    pred_top1 = int(best_cand)
                else:
                    pred_top1 = int(graph_top1)
            else:
                same += 1

        if not graph_found:
            # Fallback (graph missing): rule-only top1
            if cand_scores:
                pred_top1 = int(max(cand_scores.items(), key=lambda x: float(x[1]))[0])
            else:
                pred_top1 = -1

        # ---- breakdown counts ----
        if cand_size == 0:
            no_cand += 1
        else:
            if pred_top1 == gt:
                hit_top1 += 1
            elif gt_rank_in_cands is None:
                no_hit += 1
            else:
                hit_not_top1 += 1
                try:
                    hit_not_top1_ranks.append(int(gt_rank_in_cands))
                except Exception:
                    pass

        if graph_found and graph_top1 >= 0:
            graph_correct = (int(graph_top1) == gt)
            fused_correct = (int(pred_top1) == gt)
            if graph_correct and (not fused_correct):
                harm += 1
            elif (not graph_correct) and fused_correct:
                improve += 1
            else:
                same += 1

        # ---- relation diag accumulators ----
        n_by_rel[r] += 1
        cand_sizes_by_rel[r].append(cand_size)
        if gt_rank_in_cands is None:
            nohit_by_rel[r] += 1
        else:
            hit_by_rel[r] += 1
        if pred_top1 == gt:
            top1_hit_by_rel[r] += 1
        else:
            wrong_top1_by_rel[r] += 1
            if pred_top1 >= 0:
                wrong_top1_entities_by_rel[r][int(pred_top1)] += 1
            if gt_rank_in_cands is not None:
                ans_rank_not_top1_by_rel[r].append(int(gt_rank_in_cands))

    ranks_sorted = sorted(hit_not_top1_ranks)
    rank_stats = {
        "count": int(len(ranks_sorted)),
        "min": int(ranks_sorted[0]) if ranks_sorted else None,
        "p50": percentile_from_sorted([float(x) for x in ranks_sorted], 50),
        "p90": percentile_from_sorted([float(x) for x in ranks_sorted], 90),
        "p95": percentile_from_sorted([float(x) for x in ranks_sorted], 95),
        "p99": percentile_from_sorted([float(x) for x in ranks_sorted], 99),
        "max": int(ranks_sorted[-1]) if ranks_sorted else None,
        "mean": float(sum(ranks_sorted) / len(ranks_sorted)) if ranks_sorted else None,
    }

    denom = float(n) if n else 1.0
    breakdown_payload = {
        "meta": {
            "dataset": dataset,
            "split": split,
            "graph_reasoning_type": graph_reasoning_type,
            "rule_weight": float(rule_weight),
            "trace_enabled": False,
        },
        "N": int(n),
        "NoCand": float(no_cand / denom),
        "NoHit": float(no_hit / denom),
        "HitTop1": float(hit_top1 / denom),
        "HitNotTop1": float(hit_not_top1 / denom),
        "HitNotTop1_rank_stats": rank_stats,
        "harm_rate": float(harm / denom) if score_numpy is not None else 0.0,
    }

    rows_json: List[Dict[str, Any]] = []
    for rel, nn in n_by_rel.items():
        if int(nn) < int(min_n):
            continue

        hit_n = int(hit_by_rel.get(rel, 0))
        top1_hit_n = int(top1_hit_by_rel.get(rel, 0))
        wrong_n = int(wrong_top1_by_rel.get(rel, 0))
        nohit_n = int(nohit_by_rel.get(rel, 0))

        nohit_rate = float(nohit_n / nn) if nn else 0.0
        hit_rate = float(hit_n / nn) if nn else 0.0
        cond_top1 = float(top1_hit_n / hit_n) if hit_n else 0.0
        wrong_rate = float(wrong_n / nn) if nn else 0.0

        cand_sizes = sorted([float(x) for x in cand_sizes_by_rel.get(rel, [])])
        cand_p95 = percentile_from_sorted(cand_sizes, 95)

        ans_ranks = sorted([float(x) for x in ans_rank_not_top1_by_rel.get(rel, [])])
        ans_rank_p95 = percentile_from_sorted(ans_ranks, 95)

        top_wrong_entities = topk_counter(
            {int(e): int(c) for e, c in wrong_top1_entities_by_rel.get(rel, Counter()).items()},
            10,
        )

        rel_name = getattr(grapher, "id2relation", {}).get(int(rel), str(rel))
        worst_score = float(nohit_rate + wrong_rate)

        rows_json.append(
            {
                "rel_id": int(rel),
                "rel_name": rel_name,
                "n_test": int(nn),
                "NoHit_rate_test": float(nohit_rate),
                "Hit_rate_test": float(hit_rate),
                "CondTop1_given_Hit": float(cond_top1),
                "WrongTop1_total": int(wrong_n),
                "WrongTop1_rate": float(wrong_rate),
                "Cand_p95": float(cand_p95) if cand_p95 is not None else None,
                "AnsRankNotTop1_p95": float(ans_rank_p95) if ans_rank_p95 is not None else None,
                "TopWrongTop1Entities": top_wrong_entities,
                "_worst_score": worst_score,
            }
        )

    rows_json.sort(key=lambda x: (-float(x.get("_worst_score", 0.0)), -int(x.get("n_test", 0)), int(x.get("rel_id", 0))))
    rel_diag_payload = {
        "meta": {
            "dataset": dataset,
            "split": split,
            "graph_reasoning_type": graph_reasoning_type,
            "rule_weight": float(rule_weight),
            "trace_enabled": False,
            "min_n": int(min_n),
        },
        "relations": [{k: v for k, v in r.items() if k != "_worst_score"} for r in rows_json],
    }

    return breakdown_payload, rel_diag_payload


def _build_heuristic_harmful_rules(
    concrete_path: str,
    *,
    worst_rel_ids: List[int],
    max_rules: int,
    per_rel_top: int = 10,
) -> List[Dict[str, Any]]:
    payload = load_json(concrete_path, default=None)
    if not isinstance(payload, dict) or not worst_rel_ids:
        return []

    picked: Dict[str, Dict[str, Any]] = {}
    for rel in worst_rel_ids:
        rules = payload.get(str(int(rel))) or payload.get(int(rel))
        if not isinstance(rules, list) or not rules:
            continue

        rules_sorted = sorted(
            [r for r in rules if isinstance(r, dict) and r.get("rule_id")],
            key=lambda r: (-float(r.get("confidence", r.get("conf", 0.0))), -int(r.get("supp", 0)), str(r.get("rule_id"))),
        )
        for r in rules_sorted[: int(per_rel_top)]:
            rid = str(r.get("rule_id"))
            if rid in picked:
                continue
            conf = float(r.get("confidence", r.get("conf", 0.0)) or 0.0)
            supp = int(r.get("supp", 0) or 0)
            body_supp = int(r.get("body_supp", r.get("rule_supp", 0)) or 0)
            rule_len = int(len(r.get("body_rels") or [])) if (r.get("body_rels") is not None) else None

            low_supp_boost = 1.0 + 1.0 / math.sqrt(float(supp) + 1.0)
            len_boost = 1.0 + (0.1 * max(0, int(rule_len or 1) - 1))
            risk_score = float(conf) * float(low_supp_boost) * float(len_boost)

            picked[rid] = {
                "rule_id": rid,
                "abstract_rule": str(r.get("abstract_rule", "") or ""),
                "heuristic": True,
                "risk_score": float(round(risk_score, 6)),
                "confidence": float(conf),
                "supp": int(supp),
                "body_supp": int(body_supp),
                "rule_len": int(rule_len) if rule_len is not None else None,
                "head_rel": int(r.get("head_rel", rel)),
                "head_rel_name": str(r.get("head_rel_name", "") or ""),
            }
            if len(picked) >= int(max_rules):
                break
        if len(picked) >= int(max_rules):
            break

    items = list(picked.values())
    items.sort(
        key=lambda x: (
            -float(x.get("risk_score", 0.0)),
            -float(x.get("confidence", 0.0)),
            int(x.get("supp", 0)),
            str(x.get("rule_id", "")),
        )
    )
    return items[: int(max_rules)]


def _pick_worst_relations(rel_diag_payload: Dict[str, Any], popbias_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rels = (rel_diag_payload or {}).get("relations") or []
    pop_map = {}
    if popbias_payload:
        for r in popbias_payload.get("relations", []) or []:
            pop_map[int(r.get("rel_id", -1))] = r

    worst = []
    for r in rels:
        rel_id = int(r.get("rel_id", -1))
        item = {
            "rel_id": rel_id,
            "rel_name": r.get("rel_name", ""),
            "n": int(r.get("n_test", 0)),
            "NoHit": float(r.get("NoHit_rate_test", 0.0)),
            "CondTop1_given_Hit": float(r.get("CondTop1_given_Hit", 0.0)),
            "WrongTop1_rate": float(r.get("WrongTop1_rate", 0.0)),
            "Cand_p95": r.get("Cand_p95", None),
            "AnsRankNotTop1_p95": r.get("AnsRankNotTop1_p95", None),
            "TopWrongTop1Entities": r.get("TopWrongTop1Entities", []) or [],
        }
        if rel_id in pop_map:
            item["popbias"] = pop_map[rel_id]
        worst.append(item)

    # rel_diag is already sorted worst-first; keep top 30
    return worst[:30]


def _build_harmful_rules_stats(trace_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns rule_id -> stats dict.
    """
    stats: Dict[str, Dict[str, Any]] = {}

    def get(rid: str) -> Dict[str, Any]:
        if rid not in stats:
            stats[rid] = {
                "trigger_top1_total": 0,
                "trigger_top1_wrong": 0,
                "help_top1": 0,
                "ans_rank_sum": 0.0,
                "ans_rank_cnt": 0,
                "rel_counter": Counter(),
            }
        return stats[rid]

    iterator = iter_jsonl(trace_path)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Trace stats (rules)", unit="lines", mininterval=0.5)
    for row in iterator:
        rel = int(row.get("relation", -1))
        pred = int(row.get("pred_top1", -1))
        gt = int(row.get("gt_tail", -1))
        wrong = pred != gt
        gt_rank_in_cands = row.get("gt_rank_in_cands", None)

        for e in row.get("top1_rules", []) or []:
            rid = str(e.get("rule_id", ""))
            if not rid or rid == "GRAPH":
                continue
            s = get(rid)
            s["trigger_top1_total"] += 1
            if wrong:
                s["trigger_top1_wrong"] += 1
            s["rel_counter"][rel] += 1

        for e in row.get("ans_rules", []) or []:
            rid = str(e.get("rule_id", ""))
            if not rid or rid == "GRAPH":
                continue
            s = get(rid)
            if not wrong:
                s["help_top1"] += 1
            s["rel_counter"][rel] += 1
            if gt_rank_in_cands is not None:
                try:
                    s["ans_rank_sum"] += float(gt_rank_in_cands)
                    s["ans_rank_cnt"] += 1
                except Exception:
                    pass

    return stats


def _sample_cases(trace_path: str, worst_rel_ids: List[int], per_rel: int = 3) -> List[Dict[str, Any]]:
    remaining = {int(r): int(per_rel) for r in worst_rel_ids}
    cases: List[Dict[str, Any]] = []

    # Prefer error cases first.
    for pass_id in (1, 2):
        if not remaining:
            break
        iterator = iter_jsonl(trace_path)
        if tqdm is not None:
            iterator = tqdm(iterator, desc=f"Trace sampling (pass {pass_id})", unit="lines", mininterval=0.5)
        for row in iterator:
            rel = int(row.get("relation", -1))
            if rel not in remaining:
                continue

            pred = int(row.get("pred_top1", -1))
            gt = int(row.get("gt_tail", -1))
            gt_rank = row.get("gt_rank_in_cands", None)
            cand_size = int(row.get("cand_size", 0))

            is_error = (pred != gt) or (gt_rank is None) or (cand_size == 0)
            if (pass_id == 1) and (not is_error):
                continue
            if (pass_id == 2) and is_error:
                continue

            cases.append(
                {
                    "qid": int(row.get("qid", -1)),
                    "relation": rel,
                    "query": row.get("query", []),
                    "gt_tail": gt,
                    "pred_top1": pred,
                    "gt_rank_in_cands": gt_rank,
                    "cand_size": cand_size,
                    "top1_rules": row.get("top1_rules", []) or [],
                    "ans_rules": row.get("ans_rules", []) or [],
                }
            )
            remaining[rel] -= 1
            if remaining[rel] <= 0:
                remaining.pop(rel, None)
                if not remaining:
                    break
        # reset iterator for the next pass by reopening file: iter_jsonl does this already via generator

    return cases


def main():
    configure_stdout_utf8()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, type=str)
    parser.add_argument("--round", default=0, type=int)
    parser.add_argument("--test_data", default="valid", choices=["valid", "test"])
    parser.add_argument("--graph_reasoning_type", default="TiRGN", type=str)
    parser.add_argument("--rule_weight", default=0.9, type=float)
    parser.add_argument("--candidates_file", required=True, type=str)

    parser.add_argument("--trace_jsonl", default="", type=str)
    parser.add_argument("--breakdown_json", default="", type=str)
    parser.add_argument("--rel_diag_json", default="", type=str)
    parser.add_argument("--popbias_json", default="", type=str)
    parser.add_argument("--confidence_concrete_file", default="", type=str)

    parser.add_argument("--max_worst_relations", default=30, type=int)
    parser.add_argument("--max_harmful_rules", default=200, type=int)
    parser.add_argument("--cases_per_relation", default=3, type=int)

    parser.add_argument("--bat_file_name", type=str, default="bat_file")
    parser.add_argument("--results_root_path", type=str, default="results")
    args = parser.parse_args()

    args.results_root_path = args.results_root_path.strip('"')
    ranked_rules_dir = get_ranked_rules_dir(args.results_root_path, args.bat_file_name, args.dataset)
    names = _default_file_names(args.graph_reasoning_type, args.rule_weight, args.candidates_file)

    # Resolve candidates path (relative to ranked_rules_dir by default)
    cand_in = str(args.candidates_file).strip().strip('"')
    cand_path = cand_in if os.path.isabs(cand_in) else os.path.join(ranked_rules_dir, cand_in)
    cand_path = maybe_windows_long_path(cand_path)

    # Trace is OPTIONAL (used in A1/A3). If not provided and file doesn't exist, run aggregate-only mode (A2_no_trace).
    trace_name = str(args.trace_jsonl or "").strip().strip('"')
    trace_guess = maybe_windows_long_path(os.path.join(ranked_rules_dir, names["trace_jsonl"]))
    if trace_name:
        trace_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, trace_name))
        trace_enabled = True
    elif os.path.exists(trace_guess):
        trace_path = trace_guess
        trace_enabled = True
    else:
        trace_path = trace_guess
        trace_enabled = False

    breakdown_payload: Dict[str, Any] = {}
    rel_diag_payload: Dict[str, Any] = {}
    popbias_payload: Optional[Dict[str, Any]] = None
    notes = ""

    if trace_enabled:
        breakdown_path = maybe_windows_long_path(
            os.path.join(ranked_rules_dir, args.breakdown_json.strip() or names["breakdown_json"])
        )
        rel_diag_path = maybe_windows_long_path(
            os.path.join(ranked_rules_dir, args.rel_diag_json.strip() or names["rel_diag_json"])
        )
        popbias_path = maybe_windows_long_path(
            os.path.join(ranked_rules_dir, args.popbias_json.strip() or names["popbias_json"])
        )

        if not os.path.exists(trace_path):
            raise FileNotFoundError(f"trace jsonl not found: {trace_path}")

        breakdown_payload = load_json(breakdown_path, default={}) or {}
        rel_diag_payload = load_json(rel_diag_path, default={}) or {}
        popbias_payload = load_json(popbias_path, default=None)
    else:
        # Aggregate-only evidence (A2_no_trace): compute summary + relation stats directly from candidates + graph baseline.
        dataset_dir = maybe_windows_long_path(os.path.join("datasets", args.dataset))
        grapher = Grapher(dataset_dir)
        split_np = grapher.test_idx if args.test_data == "test" else grapher.valid_idx
        other_answers_map = _build_other_answers_map(split_np)

        baseline = _load_graph_baseline(args.dataset, dataset_dir, args.graph_reasoning_type, args.test_data)
        if baseline is not None:
            score_numpy, test_index = baseline
        else:
            score_numpy = None
            test_index = None

        candidates = _load_candidates_dict(cand_path)
        breakdown_payload, rel_diag_payload = _compute_breakdown_and_rel_diag_no_trace(
            dataset=args.dataset,
            split=args.test_data,
            graph_reasoning_type=args.graph_reasoning_type,
            rule_weight=float(args.rule_weight),
            candidates=candidates,
            split_np=split_np,
            score_numpy=score_numpy,
            test_index=test_index,
            other_answers_map=other_answers_map,
            grapher=grapher,
            min_n=50,
        )
        popbias_payload = None
        notes = "no_trace: cases/top1_rules/ans_rules removed; harmful_rules are heuristic (worst_relations-based)."

    # confidence_concrete mapping (rule_id -> abstract_rule)
    concrete_file = args.confidence_concrete_file.strip()
    if not concrete_file:
        guess1 = f"confidence_concrete_{round_tag(int(args.round))}.json"
        guess2 = f"round{int(args.round)}_confidence_concrete.json"
        guess_paths = [
            maybe_windows_long_path(os.path.join(ranked_rules_dir, guess1)),
            maybe_windows_long_path(os.path.join(ranked_rules_dir, guess2)),
            maybe_windows_long_path(os.path.join(ranked_rules_dir, "confidence_concrete.json")),
        ]
        concrete_path = None
        for gp in guess_paths:
            if os.path.exists(gp):
                concrete_path = gp
                break
        if concrete_path is None:
            concrete_path = guess_paths[-1]
    else:
        concrete_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, concrete_file))
    rule_abstract_map = _load_confidence_concrete_rule_map(concrete_path)

    worst_relations = _pick_worst_relations(rel_diag_payload, popbias_payload)[: int(args.max_worst_relations)]
    worst_rel_ids = [int(x.get("rel_id", -1)) for x in worst_relations]

    harmful_rules: List[Dict[str, Any]] = []
    cases: List[Dict[str, Any]] = []
    if trace_enabled:
        rule_stats = _build_harmful_rules_stats(trace_path)
        for rid, st in rule_stats.items():
            total = int(st.get("trigger_top1_total", 0))
            wrong = int(st.get("trigger_top1_wrong", 0))
            help_top1 = int(st.get("help_top1", 0))
            harm_rate = float(wrong / total) if total else 0.0
            ans_cnt = int(st.get("ans_rank_cnt", 0))
            avg_gt_rank = float(st.get("ans_rank_sum", 0.0) / ans_cnt) if ans_cnt else None
            rel_counter = st.get("rel_counter", Counter())
            example_rels = [int(r) for r, _ in rel_counter.most_common(10)]

            harmful_rules.append(
                {
                    "rule_id": str(rid),
                    "abstract_rule": rule_abstract_map.get(str(rid), ""),
                    "trigger_top1": int(total),
                    "harm_rate": float(harm_rate),
                    "help_top1": int(help_top1),
                    "avg_gt_rank_when_hit": avg_gt_rank,
                    "example_relations": example_rels,
                    "_wrong": wrong,
                }
            )

        harmful_rules.sort(
            key=lambda x: (-float(x.get("harm_rate", 0.0)), -int(x.get("_wrong", 0)), -int(x.get("trigger_top1", 0)))
        )
        harmful_rules = [
            {k: v for k, v in r.items() if k != "_wrong"} for r in harmful_rules[: int(args.max_harmful_rules)]
        ]

        cases = _sample_cases(trace_path, worst_rel_ids, per_rel=int(args.cases_per_relation))
    else:
        harmful_rules = _build_heuristic_harmful_rules(
            concrete_path,
            worst_rel_ids=worst_rel_ids,
            max_rules=int(args.max_harmful_rules),
            per_rel_top=10,
        )
        cases = []

    summary = {
        "N": int(breakdown_payload.get("N", 0)),
        "NoCand": float(breakdown_payload.get("NoCand", 0.0)),
        "NoHit": float(breakdown_payload.get("NoHit", 0.0)),
        "HitTop1": float(breakdown_payload.get("HitTop1", 0.0)),
        "HitNotTop1": float(breakdown_payload.get("HitNotTop1", 0.0)),
        "HitNotTop1_rank_stats": breakdown_payload.get("HitNotTop1_rank_stats", {}) or {},
        "harm_rate": float(breakdown_payload.get("harm_rate", 0.0)),
    }

    evidence = {
        "meta": {
            "dataset": args.dataset,
            "split": args.test_data,
            "graph_reasoning_type": args.graph_reasoning_type,
            "rule_weight": float(args.rule_weight),
            "round": int(args.round),
            "trace_enabled": bool(trace_enabled),
            "evidence_level": "trace" if trace_enabled else "aggregate_only",
        },
        "summary": summary,
        "worst_relations": worst_relations,
        "harmful_rules": harmful_rules,
        "cases": cases,
        "notes": notes,
        "allowed_actions": {
            "delete": True,
            "downweight": {"min_scale": 0.2, "max_scale": 0.9},
            "promote": {"max_scale": 1.2},
            "tighten_time": False,
        },
    }

    run_id = candidates_run_id(args.candidates_file)
    w_str = fmt_float_for_name(args.rule_weight, decimals=2)
    out_name = f"evidence_for_llm_{run_id}_{args.graph_reasoning_type}_w{w_str}.json"
    out_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, out_name))
    safe_dump_json(out_path, evidence, indent=2)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
