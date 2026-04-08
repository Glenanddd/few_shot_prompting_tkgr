import os.path
import json
import sys
import time
import argparse
import itertools
import shutil
import threading
from multiprocessing import Manager
import numpy as np
import torch
from joblib import Parallel, delayed

import rule_application as ra
from grapher import Grapher
from temporal_walk import store_edges
from rule_learning import rules_statistics
from score_functions import score_12, score_13, score_14
from utils import get_win_subgraph, load_json_data
from utils_windows_long_path import maybe_windows_long_path, safe_open

from params import str_to_bool

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _configure_stdout_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def parse_arguments():
    global parsed
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="", type=str)
    parser.add_argument("--test_data", default="test", type=str)
    parser.add_argument(
        "--query_dir",
        default="both",
        type=str,
        choices=["both", "forward"],
        help="Query direction to evaluate: both (default) includes inverse queries; forward evaluates only non-inverse relations.",
    )
    parser.add_argument("--rules_confidence_file", "-r", default="", type=str)
    parser.add_argument("--rule_lengths", "-l", default=1, type=int, nargs="+")
    parser.add_argument("--window", "-w", default=-1, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--num_processes", "-p", default=1, type=int)
    parser.add_argument("--rule_files", "-f", default="", type=str)
    parser.add_argument("--confidence_type", default="Common", type=str,
                        choices=['Common', 'LLM', 'And', 'Or'])
    parser.add_argument("--weight", default=0.0, type=float)
    parser.add_argument("--weight_0", default=0.5, type=float)
    parser.add_argument("--min_conf", default=0.01, type=float)
    parser.add_argument("--coor_weight", default=0, type=float)
    parser.add_argument("--lmbda", default=0.1, type=float)
    parser.add_argument("--bgkg", default="all", type=str,
                        choices=['all', 'train', 'valid', 'test', 'train_valid', 'train_test', 'valid_test'])
    parser.add_argument("--score_type", default="noisy-or", type=str,
                        choices=['noisy-or', 'sum', 'mean', 'min', 'max'])
    parser.add_argument("--is_relax_time", default='no', type=str_to_bool)
    parser.add_argument("--is_sorted", default='Yes', type=str_to_bool)
    parser.add_argument("--is_return_timestamp", default='no', type=str_to_bool)
    parser.add_argument('--evaluation_type', type=str, default='origin', choices=['transformer', 'origin', 'end'])
    parser.add_argument("--win_start", default=0, type=int)
    parser.add_argument("--is_sampled", default='no', type=str_to_bool)
    parser.add_argument("--is_rule_priority", default='no', type=str_to_bool)

    # VLRG trace (valid/test attribution)
    parser.add_argument("--trace_rules", default='No', type=str_to_bool, help="Yes|No (default No)")
    parser.add_argument("--trace_top_rules", default=5, type=int, help="Top-K rules per entity (default 5)")
    parser.add_argument("--graph_reasoning_type", default="TiRGN", type=str, help="Graph baseline type (for trace)")
    parser.add_argument("--rule_weight", default=0.9, type=float, help="Fusion rule weight w (for trace)")
    parser.add_argument(
        "--trace_file_name",
        default="",
        type=str,
        help="Optional explicit trace JSONL filename (written under ranked_rules_dir/{dataset}/)",
    )

    # VLRG naming + incremental reasoning
    parser.add_argument(
        "--candidates_file_name",
        default="",
        type=str,
        help="Optional explicit candidates JSON filename (written under ranked_rules_dir/{dataset}/)",
    )
    parser.add_argument(
        "--timestamp_file_name",
        default="",
        type=str,
        help="Optional explicit timestamp JSON filename (written under ranked_rules_dir/{dataset}/)",
    )
    parser.add_argument(
        "--qids_file",
        default="",
        type=str,
        help="Optional JSON list file of query ids to recompute (incremental reasoning)",
    )

    parser.add_argument("--bat_file_name", type=str, default='bat_file',
                        help="Batch file name")
    parser.add_argument("--results_root_path", type=str, default='results',
                        help="Results root path. Must put this parameter at last position on cmd line to avoid parsing error.")
    return vars(parser.parse_args())


def _build_candidates_filename(rules_confidence_file, rule_lengths, window, score_func_str):
    filename = "{0}_cands_r{1}_w{2}_{3}.json".format(
        rules_confidence_file[:-11], rule_lengths, window, score_func_str
    )
    return filename.replace(" ", "")


def _load_rule_meta_from_concrete(ranked_rules_path, rules_confidence_file):
    """
    Load rule_id -> {abstract_rule, confidence} from confidence_concrete*.json.
    """
    base = os.path.basename(rules_confidence_file)
    concrete_guess = base.replace("confidence", "confidence_concrete", 1)
    concrete_path = maybe_windows_long_path(os.path.join(ranked_rules_path, concrete_guess))
    if not os.path.exists(concrete_path):
        concrete_path = maybe_windows_long_path(os.path.join(ranked_rules_path, "confidence_concrete.json"))

    payload = load_json_data(concrete_path) or {}
    rule_meta = {}
    for _, rules in payload.items():
        for r in rules or []:
            rid = r.get("rule_id")
            if not rid:
                continue
            rule_meta[str(rid)] = {
                "abstract_rule": r.get("abstract_rule", ""),
                "confidence": float(r.get("confidence", r.get("conf", 0.0))),
            }
    return rule_meta, concrete_path


def _load_graph_scores_for_trace(dataset, dataset_dir, graph_reasoning_type, split="test"):
    """
    Load graph baseline files used by evaluate.py / trace.
    - split=test:  test.npy + score.npy
    - split=valid: test_valid.npy + score_valid.npy (fallback to test.npy+score.npy)
    Note: Some setups export both valid+test queries into the same test.npy/score.npy.
    """
    split_s = str(split or "test").lower()
    base_dir = maybe_windows_long_path(os.path.join(dataset_dir, graph_reasoning_type))
    if split_s == "valid":
        pairs = [("test_valid.npy", "score_valid.npy"), ("valid.npy", "score_valid.npy"), ("test.npy", "score.npy")]
    else:
        pairs = [("test.npy", "score.npy")]
    tried = []
    test_path = None
    score_path = None
    for t_name, s_name in pairs:
        tp = maybe_windows_long_path(os.path.join(base_dir, t_name))
        sp = maybe_windows_long_path(os.path.join(base_dir, s_name))
        tried.append((tp, sp))
        if os.path.exists(tp) and os.path.exists(sp):
            test_path, score_path = tp, sp
            break
    if not test_path or not score_path:
        tried_str = "\n".join([f"  test={t} | score={s}" for t, s in tried])
        raise FileNotFoundError(
            "Missing graph baseline files required by trace.\n"
            f"split={split_s} graph={graph_reasoning_type}\n"
            "Tried:\n"
            f"{tried_str}\n"
            "Please export them first (same as evaluate.py requirement)."
        )
    test_numpy = np.load(test_path)
    if dataset == "icews18":
        test_numpy[:, 3] = (test_numpy[:, 3] / 24).astype(int)
    score_numpy = np.load(score_path, mmap_mode="r")

    test_index = {}
    for i, row in enumerate(test_numpy):
        key = tuple(int(x) for x in row)
        if key not in test_index:
            test_index[key] = i
    return test_numpy, score_numpy, test_index, test_path, score_path

def _build_other_answers_map(test_data_np):
    mp = {}
    for row in test_data_np:
        s = int(row[0])
        r = int(row[1])
        o = int(row[2])
        t = int(row[3])
        key = (s, r, t)
        mp.setdefault(key, []).append(o)
    return mp


def _topk_rule_deltas_for_entity(scores_by_rule_id, score_type, rule_weight, top_k, rule_meta):
    """
    Return a list[dict] of top-K rules by delta for ONE entity.

    Delta definition:
      - score_type == 'noisy-or': delta = w * (S(all) - S(without this rule))
      - score_type == 'sum':      delta = w * sum(scores_of_rule)
      - else:                     approximation: w * agg(scores_of_rule)

    For noisy-or, S(.) is computed on the rule-only side (before fusion).
    """
    if not scores_by_rule_id:
        return []

    entries = []

    if score_type == "noisy-or":
        s_rule = {}
        for rid, vals in scores_by_rule_id.items():
            prod = 1.0
            for v in vals:
                vv = float(v)
                if vv <= 0:
                    continue
                if vv >= 1:
                    prod = 0.0
                    break
                prod *= (1.0 - vv)
            s_rule[rid] = 1.0 - prod

        p_all = 1.0
        for v in s_rule.values():
            p_all *= (1.0 - float(v))

        for rid, s in s_rule.items():
            one_minus = 1.0 - float(s)
            if one_minus > 1e-12:
                p_other = p_all / one_minus
            else:
                # fall back: explicit product over others when one_minus == 0
                p_other = 1.0
                for rid2, s2 in s_rule.items():
                    if rid2 == rid:
                        continue
                    p_other *= (1.0 - float(s2))

            delta = float(rule_weight) * float(s) * float(p_other)
            meta = rule_meta.get(str(rid), {})
            entries.append(
                {
                    "rule_id": str(rid),
                    "delta": float(delta),
                    "abstract_rule": meta.get("abstract_rule", ""),
                    "confidence": float(meta.get("confidence", 0.0)),
                }
            )

    elif score_type == "sum":
        for rid, vals in scores_by_rule_id.items():
            s = float(sum(float(v) for v in vals))
            delta = float(rule_weight) * s
            meta = rule_meta.get(str(rid), {})
            entries.append(
                {
                    "rule_id": str(rid),
                    "delta": float(delta),
                    "abstract_rule": meta.get("abstract_rule", ""),
                    "confidence": float(meta.get("confidence", 0.0)),
                }
            )
    else:
        # Approximation for mean/min/max: treat per-rule aggregated score as "delta"
        for rid, vals in scores_by_rule_id.items():
            arr = [float(v) for v in vals]
            if not arr:
                continue
            if score_type == "mean":
                s = float(sum(arr) / max(1, len(arr)))
            elif score_type == "min":
                s = float(min(arr))
            elif score_type == "max":
                s = float(max(arr))
            else:
                s = float(sum(arr))
            delta = float(rule_weight) * s
            meta = rule_meta.get(str(rid), {})
            entries.append(
                {
                    "rule_id": str(rid),
                    "delta": float(delta),
                    "abstract_rule": meta.get("abstract_rule", ""),
                    "confidence": float(meta.get("confidence", 0.0)),
                }
            )

    entries.sort(key=lambda x: x["delta"], reverse=True)
    return entries[: int(top_k)]

def apply_rules(
    i,
    num_queries,
    parsed,
    test_data,
    windown_subgraph,
    rules_dict,
    args,
    score_func,
    top_k,
    qids_chunk=None,
    progress_counter=None,
    progress_lock=None,
    progress_update_every=100,
    trace_ctx=None,
    trace_cfg=None,
):
    """
    Apply rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_queries (int): minimum number of queries for each process

    Returns:
        all_candidates (list): answer candidates with corresponding confidence scores
        no_cands_counter (int): number of queries with no answer candidates
    """

    if progress_counter is None:
        print("Start process", i, "...")
    torch.cuda.set_device(parsed['gpu'])
    all_candidates = [dict() for _ in range(len(args))]
    all_timestamp = [dict() for _ in range(len(args))]
    no_cands_counter = 0

    if qids_chunk is not None:
        test_queries_idx = list(qids_chunk)
    else:
        test_queries_idx = (
            range(i * num_queries, (i + 1) * num_queries)
            if i < parsed['num_processes'] - 1
            else range(i * num_queries, len(test_data))
        )

    if len(test_queries_idx) == 0:
        return all_candidates, no_cands_counter, all_timestamp, 0

    # Per-process trace context (multiprocess-safe): each process writes its own part file.
    local_trace_ctx = trace_ctx
    local_trace_fout = None
    if local_trace_ctx is None and trace_cfg is not None:
        part_paths = trace_cfg.get("part_paths") or []
        if i >= len(part_paths):
            raise IndexError(f"trace_cfg.part_paths missing index {i}")
        part_path = part_paths[i]
        try:
            local_trace_fout = safe_open(part_path, "w", encoding="utf-8", errors="replace")
        except PermissionError as e:
            raise PermissionError(
                f"PermissionError writing trace part file: {part_path}\n"
                "This can happen if the file is opened in Excel or another program. "
                "Please close it or change output name."
            ) from e
        score_numpy = np.load(trace_cfg["score_path"], mmap_mode="r")
        local_trace_ctx = {
            "rule_meta": trace_cfg.get("rule_meta", {}),
            "score_numpy": score_numpy,
            "test_index": trace_cfg.get("test_index", {}),
            "other_answers_map": trace_cfg.get("other_answers_map", {}),
            "rule_weight": float(trace_cfg.get("rule_weight", 0.9)),
            "trace_top_rules": int(trace_cfg.get("trace_top_rules", 5)),
            "trace_fout": local_trace_fout,
            "trace_errors": 0,
        }

    cur_ts = test_data[test_queries_idx[0]][3]
    edges = windown_subgraph[cur_ts]

    it_start = time.time()
    local_processed = 0
    for j in test_queries_idx:
        # j_start = time.time()

        test_query = test_data[j]
        cands_dict = [dict() for _ in range(len(args))]
        timestamp_dict = [dict() for _ in range(len(args))]
        trace_scores_by_cand = {} if (local_trace_ctx is not None) else None

        if test_query[3] != cur_ts:
            cur_ts = test_query[3]
            edges = windown_subgraph[cur_ts]

        if test_query[1] in rules_dict:
            dicts_idx = list(range(len(args)))
            for rule_idx, rule in enumerate(rules_dict[test_query[1]]):
                walk_edges = ra.match_body_relations(rule, edges, test_query, is_sample=parsed["is_sampled"])

                corre = 0

                if 0 not in [len(x) for x in walk_edges]:
                    if parsed['evaluation_type'] != 'end':
                       rule_walks = ra.get_walks(rule, walk_edges, parsed["is_relax_time"])
                    else:
                       rule_walks = ra.get_walks_end(rule, walk_edges, parsed["is_relax_time"])

                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    if not rule_walks.empty:
                        cands_dict, timestamp_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            cur_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                            corre,
                            parsed['is_return_timestamp'],
                            parsed['evaluation_type'],
                            timestamp_dict,
                            trace_scores_by_cand=trace_scores_by_cand,
                            trace_rule_id=rule.get("rule_id", "UNKNOWN"),
                        )
                        for s in dicts_idx:
                            if parsed['is_return_timestamp'] is True:
                                for x in cands_dict[s].keys():
                                    # 获取 cands_dict 中排序后的列表及其索引
                                    sorted_indices, sorted_cands = zip(
                                        *sorted(enumerate(cands_dict[s][x]), key=lambda pair: pair[1],
                                                reverse=True))

                                    sorted_indices = list(sorted_indices)
                                    sorted_cands = list(sorted_cands)

                                    # 重新排列 timestamp_dict 中对应的列表
                                    timestamp_dict[s][x] = [timestamp_dict[s][x][i] for i in sorted_indices]

                                    # 更新 cands_dict 中的列表为排序后的列表
                                    cands_dict[s][x] = sorted_cands

                                # 对 cands_dict[s] 进行排序
                                sorted_items = sorted(cands_dict[s].items(), key=lambda item: item[1], reverse=True)

                                # 更新 cands_dict[s]
                                cands_dict[s] = dict(sorted_items)

                                # 使用相同的顺序更新 timestamp_dict[s]
                                timestamp_dict[s] = {k: timestamp_dict[s][k] for k, _ in sorted_items}


                            else:
                                cands_dict[s] = {
                                    x: sorted(cands_dict[s][x], reverse=True)
                                    for x in cands_dict[s].keys()
                                }

                                cands_dict[s] = dict(sorted(cands_dict[s].items(),key=lambda item: item[1], reverse=True))

                            if parsed['is_rule_priority'] is False:
                                top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                                unique_scores = list(
                                    scores for scores, _ in itertools.groupby(top_k_scores)
                                )
                                if len(unique_scores) >= top_k:
                                    dicts_idx.remove(s)
                            else:
                                if rule_idx >= top_k:
                                    dicts_idx.remove(s)

                        if not dicts_idx:
                            break

            if cands_dict[0]:
                for s in range(len(args)):
                    # Calculate noisy-or scores
                    scores = calculate_scores(cands_dict[s], parsed)

                    scores = [np.float64(x) for x in scores]
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                    )

                    temp_timestamp = {}
                    if parsed['is_return_timestamp'] is True:
                        for time_key, time_value in timestamp_dict[s].items():
                            temp_timestamp[time_key] = max(time_value)
                        all_timestamp[s][j] = temp_timestamp

                    all_candidates[s][j] = noisy_or_cands
            else:  # No candidates found by applying rules
                no_cands_counter += 1
                for s in range(len(args)):
                    all_candidates[s][j] = dict()
                    if parsed['is_return_timestamp'] is True:
                       all_timestamp[s][j] = dict()

        else:  # No rules exist for this relation
            no_cands_counter += 1
            for s in range(len(args)):
                all_candidates[s][j] = dict()
                if parsed['is_return_timestamp'] is True:
                    all_timestamp[s][j] = dict()

        # ---- VLRG trace ----
        if local_trace_ctx is not None:
            rule_meta = local_trace_ctx["rule_meta"]
            score_numpy = local_trace_ctx["score_numpy"]
            test_index = local_trace_ctx["test_index"]
            other_answers_map = local_trace_ctx["other_answers_map"]
            graph_w = float(local_trace_ctx["rule_weight"])
            top_k_rules = int(local_trace_ctx["trace_top_rules"])
            trace_fout = local_trace_ctx["trace_fout"]

            s_id = int(test_query[0])
            r_id = int(test_query[1])
            ans = int(test_query[2])
            ts = int(test_query[3])

            cand_scores = all_candidates[0].get(j, {}) or {}
            cand_size = int(len(cand_scores))

            gt_rank_in_cands = None
            if ans in cand_scores:
                all_confs = sorted(list(cand_scores.values()), reverse=True)
                try:
                    gt_rank_in_cands = int(all_confs.index(cand_scores[ans]) + 1)
                except ValueError:
                    gt_rank_in_cands = None

            trace_line = {
                "qid": int(j),
                "query": [int(x) for x in test_query],
                "relation": int(r_id),
                "gt_tail": int(ans),
                "pred_top1": int(-1),
                "gt_rank_in_cands": gt_rank_in_cands,
                "cand_size": cand_size,
                "top1_rules": [],
                "ans_rules": [],
            }

            try:
                key = tuple(int(x) for x in test_query)
                pos = test_index.get(key)
                graph_found = pos is not None

                if graph_found:
                    regcn_score = np.array(score_numpy[pos], dtype=np.float32, copy=True)
                else:
                    # Fallback: graph baseline missing for this query; treat graph score as zeros.
                    regcn_score = np.zeros(int(score_numpy.shape[1]), dtype=np.float32)

                fused = regcn_score * (1.0 - graph_w)
                for eid, rs in cand_scores.items():
                    fused[int(eid)] += graph_w * float(rs)

                # filter other answers (keep ans)
                objs = other_answers_map.get((s_id, r_id, ts), [])
                if objs:
                    for o in objs:
                        if int(o) != ans:
                            fused[int(o)] = -np.inf

                if graph_found:
                    pred_top1 = int(np.argmax(fused))
                else:
                    # When graph is missing, use rule-only top1 (avoid argmax over all-zero vector).
                    pred_top1 = int(max(cand_scores.items(), key=lambda x: x[1])[0]) if cand_scores else -1

                # Rules attribution (top-K). If entity has no rule contribution, mark source as GRAPH/GRAPH_MISSING.
                if pred_top1 >= 0 and trace_scores_by_cand and pred_top1 in trace_scores_by_cand:
                    top1_rules = _topk_rule_deltas_for_entity(
                        scores_by_rule_id=trace_scores_by_cand.get(pred_top1, {}),
                        score_type=parsed["score_type"],
                        rule_weight=graph_w,
                        top_k=top_k_rules,
                        rule_meta=rule_meta,
                    )
                else:
                    top1_rules = [
                        {
                            "rule_id": "GRAPH" if graph_found else "GRAPH_MISSING",
                            "delta": float((1.0 - graph_w) * float(regcn_score[pred_top1])) if (graph_found and pred_top1 >= 0) else 0.0,
                            "abstract_rule": "GRAPH" if graph_found else "GRAPH_MISSING",
                            "confidence": float(regcn_score[pred_top1]) if (graph_found and pred_top1 >= 0) else 0.0,
                        }
                    ]

                if trace_scores_by_cand and ans in trace_scores_by_cand:
                    ans_rules = _topk_rule_deltas_for_entity(
                        scores_by_rule_id=trace_scores_by_cand.get(ans, {}),
                        score_type=parsed["score_type"],
                        rule_weight=graph_w,
                        top_k=top_k_rules,
                        rule_meta=rule_meta,
                    )
                else:
                    ans_rules = [
                        {
                            "rule_id": "GRAPH" if graph_found else "GRAPH_MISSING",
                            "delta": float((1.0 - graph_w) * float(regcn_score[ans])) if graph_found else 0.0,
                            "abstract_rule": "GRAPH" if graph_found else "GRAPH_MISSING",
                            "confidence": float(regcn_score[ans]) if graph_found else 0.0,
                        }
                    ]

                trace_line["pred_top1"] = int(pred_top1)
                trace_line["top1_rules"] = top1_rules
                trace_line["ans_rules"] = ans_rules
                trace_line["graph_found"] = bool(graph_found)
            except Exception as e:
                local_trace_ctx["trace_errors"] = local_trace_ctx.get("trace_errors", 0) + 1
                trace_line["trace_error"] = str(e)[:200]
                if not trace_line["top1_rules"]:
                    trace_line["top1_rules"] = [
                        {"rule_id": "TRACE_ERROR", "delta": 0.0, "abstract_rule": "TRACE_ERROR", "confidence": 0.0}
                    ]
                if not trace_line["ans_rules"]:
                    trace_line["ans_rules"] = [
                        {"rule_id": "TRACE_ERROR", "delta": 0.0, "abstract_rule": "TRACE_ERROR", "confidence": 0.0}
                    ]

            trace_fout.write(json.dumps(trace_line, ensure_ascii=False) + "\n")


        if progress_counter is not None:
            local_processed += 1
            if local_processed >= progress_update_every:
                if progress_lock is None:
                    progress_counter.value += local_processed
                else:
                    with progress_lock:
                        progress_counter.value += local_processed
                local_processed = 0
        elif not (j - test_queries_idx[0] + 1) % 100:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test samples finished: {1}/{2}, {3} sec".format(
                    i, j - test_queries_idx[0] + 1, len(test_queries_idx), it_time
                )
            )
            it_start = time.time()

    if progress_counter is not None and local_processed:
        if progress_lock is None:
            progress_counter.value += local_processed
        else:
            with progress_lock:
                progress_counter.value += local_processed

    trace_errors = int(local_trace_ctx.get("trace_errors", 0)) if local_trace_ctx is not None else 0
    if local_trace_fout is not None:
        try:
            local_trace_fout.close()
        except Exception:
            pass
    return all_candidates, no_cands_counter, all_timestamp, trace_errors


def calculate_scores(cands_dict, parsed):
    if parsed["score_type"] == 'noisy-or':
        scores = list(
            map(
                lambda x: 1 - np.prod(1 - np.array(x)),
                cands_dict.values(),
            )
        )
    elif parsed['score_type'] == 'sum':
        scores = [sum(sublist) for sublist in list(cands_dict.values())]
    elif parsed['score_type'] == 'mean':
        scores = [sum(sublist) / len(sublist) for sublist in list(cands_dict.values())]
    elif parsed['score_type'] == 'min':
        scores = [min(sublist) for sublist in list(cands_dict.values())]
    elif parsed['score_type'] == 'max':
        scores = [max(sublist) for sublist in list(cands_dict.values())]
    return scores

def load_rules(rules_file, dir_path):
    rules_dict = load_json_data(maybe_windows_long_path(os.path.join(dir_path, rules_file)))
    return {int(k): v for k, v in rules_dict.items()}
def get_score_func(parsed):
    if parsed['evaluation_type'] == 'origin':
        parsed['coor_weight'] = 0
        return score_12
    elif parsed['evaluation_type'] == 'transformer':
        return score_13
    elif parsed['evaluation_type'] == 'end':
        parsed['coor_weight'] = 0
        return score_14

def apply_rules_in_parallel(parsed, test_data, windown_subgraph, rules_dict, args, score_func, trace_cfg=None, qids=None):
    final_all_candidates = [dict() for _ in range(len(args))]
    final_all_timestamp = [dict() for _ in range(len(args))]
    final_no_cands_counter = 0
    final_trace_errors = 0

    start = time.time()

    progress_counter = None
    progress_lock = None
    progress_thread = None
    progress_stop = None
    progress_bar = None
    total_queries = len(qids) if qids is not None else len(test_data)
    progress_update_every = 100

    if tqdm is not None and total_queries > 0:
        manager = Manager()
        progress_counter = manager.Value("i", 0)
        progress_lock = manager.Lock()
        progress_update_every = max(1, min(100, total_queries // 100))

        progress_bar = tqdm(
            total=total_queries,
            desc="Overall query progress",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        progress_stop = threading.Event()

        def monitor_progress():
            last = 0
            while not progress_stop.is_set():
                try:
                    with progress_lock:
                        current = progress_counter.value
                except Exception:
                    # If the Manager connection is broken (e.g., worker crash), stop monitoring quietly.
                    break
                if current > last:
                    progress_bar.update(current - last)
                    last = current
                time.sleep(0.2)
            try:
                with progress_lock:
                    current = progress_counter.value
            except Exception:
                return
            if current > last:
                progress_bar.update(current - last)

        progress_thread = threading.Thread(target=monitor_progress, daemon=True)
        progress_thread.start()

    qids_chunks = None
    if qids is not None:
        qids_s = []
        for x in qids:
            try:
                xi = int(x)
            except Exception:
                continue
            if 0 <= xi < len(test_data):
                qids_s.append(xi)
        qids_s = sorted(set(qids_s))
        nproc = int(parsed["num_processes"])
        base = len(qids_s) // nproc if nproc > 0 else 0
        rem = len(qids_s) % nproc if nproc > 0 else 0
        qids_chunks = []
        start = 0
        for i in range(nproc):
            size = base + (1 if i < rem else 0)
            end = start + size
            qids_chunks.append(qids_s[start:end])
            start = end

    num_queries = len(test_data) // parsed['num_processes']
    output = Parallel(n_jobs=parsed['num_processes'])(
        delayed(apply_rules)(
            i,
            num_queries,
            parsed,
            test_data,
            windown_subgraph,
            rules_dict,
            args,
            score_func,
            parsed['top_k'],
            qids_chunks[i] if qids_chunks is not None else None,
            progress_counter,
            progress_lock,
            progress_update_every,
            None,
            trace_cfg,
        )
        for i in range(parsed['num_processes'])
    )

    for s in range(len(args)):
        for i in range(parsed['num_processes']):
            final_all_candidates[s].update(output[i][0][s])
            output[i][0][s].clear()

            final_all_timestamp[s].update(output[i][2][s])
            output[i][2][s].clear()

    for i in range(parsed['num_processes']):
        final_no_cands_counter += output[i][1]
        if len(output[i]) > 3:
            final_trace_errors += int(output[i][3])

    end = time.time()
    total_time = round(end - start, 6)
    print("Application finished in {} seconds.".format(total_time))

    if progress_stop is not None:
        progress_stop.set()
    if progress_thread is not None:
        progress_thread.join()
    if progress_bar is not None:
        remaining = total_queries - progress_bar.n
        if remaining > 0:
            progress_bar.update(remaining)
        progress_bar.close()

    return final_all_candidates, final_all_timestamp, final_no_cands_counter, final_trace_errors

def print_final_statistics(final_no_cands_counter, final_all_candidates):
    print("No candidates: ", final_no_cands_counter, " queries")

def save_results(final_all_candidates, final_all_timestamp, parsed, ranked_rules_path, rules_confidence_file, args, score_func):
    saved_files = []
    explicit_cand = str(parsed.get("candidates_file_name", "") or "").strip().strip('"')
    explicit_ts = str(parsed.get("timestamp_file_name", "") or "").strip().strip('"')
    if explicit_cand:
        explicit_cand = os.path.basename(explicit_cand)
        if not explicit_cand.lower().endswith(".json"):
            explicit_cand = explicit_cand + ".json"
    if explicit_ts:
        explicit_ts = os.path.basename(explicit_ts)
    elif explicit_cand:
        explicit_ts = os.path.splitext(explicit_cand)[0] + "_timestamp.json"

    for s in range(len(args)):
        score_func_str = f'{score_func.__name__}{args[s]}'.replace(" ", "")
        score_func_str = f'{score_func_str}_rule_{parsed["is_rule_priority"]}_top_{parsed["top_k"]}_et_{parsed["evaluation_type"]}_sorted_{parsed["is_sorted"]}_bgkg_{parsed["bgkg"]}_start_{parsed["win_start"]}_relax_{parsed["is_relax_time"]}_sample_{parsed["is_sampled"]}'

        cand_name_s = explicit_cand
        ts_name_s = explicit_ts
        if explicit_cand and len(args) > 1:
            base, ext = os.path.splitext(explicit_cand)
            cand_name_s = f"{base}_s{s}{ext}"
            if explicit_ts:
                ts_base, ts_ext = os.path.splitext(explicit_ts)
                ts_name_s = f"{ts_base}_s{s}{ts_ext}"

        c_fn, t_fn = ra.save_candidates(
            rules_confidence_file,
            ranked_rules_path,
            final_all_candidates[s],
            parsed["rule_lengths"],
            parsed["window"],
            score_func_str,
            final_all_timestamp[s],
            candidates_file_name=cand_name_s,
            timestamp_file_name=ts_name_s,
        )
        saved_files.append((c_fn, t_fn))
    return saved_files

def main():
    _configure_stdout_utf8()
    parsed = parse_arguments()

    dataset_dir = maybe_windows_long_path(os.path.join(".", "datasets", parsed["dataset"]))
    
    parsed["results_root_path"] = parsed["results_root_path"].strip('"')
    ranked_rules_path = maybe_windows_long_path(os.path.join(parsed["results_root_path"], parsed["bat_file_name"], "ranked_rules", parsed["dataset"]))

    data = Grapher(dataset_dir, parsed)
    test_data = data.test_idx if parsed["test_data"] == "test" else data.valid_idx

    # Optional incremental reasoning: only recompute selected qids (still uses full test_data for indexing).
    qids = None
    qids_file = str(parsed.get("qids_file", "") or "").strip().strip('"')
    if qids_file:
        qids_path = qids_file
        if not os.path.isabs(qids_path):
            qids_path = os.path.join(ranked_rules_path, qids_path)
        qids_path = maybe_windows_long_path(qids_path)
        with safe_open(qids_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"--qids_file must be a JSON list: {qids_path}")
        qids_s = []
        for x in payload:
            try:
                xi = int(x)
            except Exception:
                continue
            if 0 <= xi < len(test_data):
                qids_s.append(xi)
        qids = sorted(set(qids_s))

    # Optional: evaluate only forward (non-inverse) queries.
    query_dir = str(parsed.get("query_dir", "both") or "both").lower().strip()
    parsed["query_dir"] = query_dir
    if query_dir == "forward":
        num_relations_old = int(len(getattr(data, "relation2id_old", {}) or {}))
        if num_relations_old <= 0:
            print("[Warn] query_dir=forward but relation2id_old missing; fall back to both.")
        else:
            forward_qids = np.nonzero(test_data[:, 1] < num_relations_old)[0].astype(int).tolist()
            if qids is None:
                qids = forward_qids
            else:
                qids = sorted(set(int(x) for x in qids).intersection(set(forward_qids)))
            print(f"[QueryDir] query_dir=forward => qids={len(qids)}/{len(test_data)}")
    ranked_rules_dict = load_rules(parsed["rules_confidence_file"], ranked_rules_path)

    print("Rules statistics:")
    rules_statistics(ranked_rules_dict)

    score_func = get_score_func(parsed)
    args = [[parsed['lmbda'], parsed['weight_0'], parsed['confidence_type'], parsed['weight'], parsed['min_conf'], parsed['coor_weight']]]

    new_rules_dict, sort_rules_dict = ra.filter_rules(
        ranked_rules_dict, min_conf=parsed['min_conf'], min_body_supp=2, rule_lengths=parsed["rule_lengths"], confidence_type=parsed["confidence_type"]
    )

    ranked_rules_dict = new_rules_dict if not parsed["is_sorted"] else sort_rules_dict

    print("Rules statistics after pruning:")
    rules_statistics(ranked_rules_dict)
    learn_edges = store_edges(data.train_idx)

    windown_subgraph = get_win_subgraph(test_data, data, learn_edges, parsed["window"], win_start=parsed["win_start"])

    trace_ctx = None
    if parsed.get("trace_rules"):
        score_func_str = f'{score_func.__name__}{args[0]}'.replace(" ", "")
        score_func_str = f'{score_func_str}_rule_{parsed["is_rule_priority"]}_top_{parsed["top_k"]}_et_{parsed["evaluation_type"]}_sorted_{parsed["is_sorted"]}_bgkg_{parsed["bgkg"]}_start_{parsed["win_start"]}_relax_{parsed["is_relax_time"]}_sample_{parsed["is_sampled"]}'
        explicit_cand = str(parsed.get("candidates_file_name", "") or "").strip().strip('"')
        if explicit_cand:
            cand_fn = os.path.basename(explicit_cand)
        else:
            cand_fn = _build_candidates_filename(parsed["rules_confidence_file"], parsed["rule_lengths"], parsed["window"], score_func_str)
        cand_base = os.path.splitext(os.path.basename(cand_fn))[0]
        run_part = cand_base[len("cands_") :] if cand_base.startswith("cands_") else cand_base
        w_str = f"{float(parsed['rule_weight']):.2f}".rstrip("0").rstrip(".")

        trace_name = str(parsed.get("trace_file_name", "") or "").strip().strip('"')
        if not trace_name:
            trace_name = f"trace_{run_part}_{parsed['graph_reasoning_type']}_w{w_str}.jsonl"
        trace_name = os.path.basename(trace_name)
        trace_path = maybe_windows_long_path(os.path.join(ranked_rules_path, trace_name))

        rule_meta, concrete_path = _load_rule_meta_from_concrete(ranked_rules_path, parsed["rules_confidence_file"])
        test_numpy, score_numpy, test_index, test_path, score_path = _load_graph_scores_for_trace(
            dataset=parsed["dataset"],
            dataset_dir=dataset_dir,
            graph_reasoning_type=parsed["graph_reasoning_type"],
            split=parsed["test_data"],
        )
        other_answers_map = _build_other_answers_map(test_data)

        os.makedirs(ranked_rules_path, exist_ok=True)

        num_p = int(parsed.get("num_processes", 1))
        if num_p <= 1:
            try:
                trace_fout = safe_open(trace_path, "w", encoding="utf-8", errors="replace")
            except PermissionError as e:
                raise PermissionError(
                    f"PermissionError writing trace file: {trace_path}\n"
                    "This can happen if the file is opened in Excel or another program. "
                    "Please close it or change output name."
                ) from e

            trace_ctx = {
                "rule_meta": rule_meta,
                "concrete_path": concrete_path,
                "score_numpy": score_numpy,
                "test_index": test_index,
                "other_answers_map": other_answers_map,
                "rule_weight": float(parsed["rule_weight"]),
                "trace_top_rules": int(parsed.get("trace_top_rules", 5)),
                "trace_fout": trace_fout,
                "trace_errors": 0,
            }

            final_all_candidates, final_no_cands_counter, final_all_timestamp, trace_errors = apply_rules(
                i=0,
                num_queries=len(test_data),
                parsed=parsed,
                test_data=test_data,
                windown_subgraph=windown_subgraph,
                rules_dict=ranked_rules_dict,
                args=args,
                score_func=score_func,
                top_k=parsed["top_k"],
                qids_chunk=qids,
                trace_ctx=trace_ctx,
            )
            trace_fout.close()
        else:
            part_paths = [
                maybe_windows_long_path(trace_path.replace(".jsonl", f".part{i}.jsonl")) for i in range(num_p)
            ]
            trace_cfg = {
                "part_paths": part_paths,
                "score_path": score_path,
                "test_index": test_index,
                "other_answers_map": other_answers_map,
                "rule_meta": rule_meta,
                "rule_weight": float(parsed["rule_weight"]),
                "trace_top_rules": int(parsed.get("trace_top_rules", 5)),
            }
            final_all_candidates, final_all_timestamp, final_no_cands_counter, trace_errors = apply_rules_in_parallel(
                parsed, test_data, windown_subgraph, ranked_rules_dict, args, score_func, trace_cfg=trace_cfg, qids=qids
            )

            # Merge parts deterministically (part0, part1, ... => stable qid order)
            try:
                with safe_open(trace_path, "w", encoding="utf-8", errors="replace") as fout:
                    for p in part_paths:
                        with safe_open(p, "r", encoding="utf-8", errors="replace") as fin:
                            shutil.copyfileobj(fin, fout)
            except PermissionError as e:
                raise PermissionError(
                    f"PermissionError writing merged trace file: {trace_path}\n"
                    "This can happen if the file is opened in Excel or another program. "
                    "Please close it or change output name."
                ) from e

            for p in part_paths:
                try:
                    os.remove(maybe_windows_long_path(p))
                except Exception:
                    pass

        if trace_errors:
            print(f"[Trace] warnings: trace_errors={trace_errors}")
    else:
        final_all_candidates, final_all_timestamp, final_no_cands_counter, _trace_errors = apply_rules_in_parallel(
            parsed, test_data, windown_subgraph, ranked_rules_dict, args, score_func, qids=qids
        )

    print_final_statistics(final_no_cands_counter, final_all_candidates)

    save_results(final_all_candidates, final_all_timestamp, parsed, ranked_rules_path,  parsed["rules_confidence_file"], args, score_func)

if __name__ == "__main__":
    main()
