import argparse
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grapher import Grapher
from utils_windows_long_path import maybe_windows_long_path
from vlrg_utils import (
    candidates_run_id,
    configure_stdout_utf8,
    fmt_float_for_name,
    get_ranked_rules_dir,
    iter_jsonl,
    percentile_from_sorted,
    safe_dump_json,
    safe_write_text,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _default_trace_name(graph_reasoning_type: str, rule_weight: float, candidates_file: str) -> str:
    cand_base = os.path.splitext(os.path.basename(candidates_file))[0]
    run_id = cand_base[len("cands_") :] if cand_base.startswith("cands_") else cand_base
    w_str = fmt_float_for_name(rule_weight, decimals=2)
    return f"trace_{run_id}_{graph_reasoning_type}_w{w_str}.jsonl"


def _load_graph_baseline(dataset: str, dataset_dir: str, graph_reasoning_type: str, split: str):
    split_s = str(split or "test").lower()
    if split_s == "valid":
        test_path = maybe_windows_long_path(os.path.join(dataset_dir, graph_reasoning_type, "test_valid.npy"))
        score_path = maybe_windows_long_path(os.path.join(dataset_dir, graph_reasoning_type, "score_valid.npy"))
        if not os.path.exists(score_path):
            typo_score_path = maybe_windows_long_path(os.path.join(dataset_dir, graph_reasoning_type, "score_vlid.npy"))
            if os.path.exists(typo_score_path):
                score_path = typo_score_path
    else:
        test_path = maybe_windows_long_path(os.path.join(dataset_dir, graph_reasoning_type, "test.npy"))
        score_path = maybe_windows_long_path(os.path.join(dataset_dir, graph_reasoning_type, "score.npy"))

    if not os.path.exists(test_path) or not os.path.exists(score_path):
        return None

    test_numpy = np.load(test_path)
    if dataset == "icews18":
        test_numpy[:, 3] = (test_numpy[:, 3] / 24).astype(int)
    score_numpy = np.load(score_path, mmap_mode="r")

    test_index = {}
    for i, row in enumerate(test_numpy):
        key = tuple(int(x) for x in row)
        if key not in test_index:
            test_index[key] = i
    return score_numpy, test_index


def _build_other_answers_map(test_data_np: np.ndarray) -> Dict[Tuple[int, int, int], List[int]]:
    mp: Dict[Tuple[int, int, int], List[int]] = {}
    for row in test_data_np:
        s = int(row[0])
        r = int(row[1])
        o = int(row[2])
        t = int(row[3])
        mp.setdefault((s, r, t), []).append(o)
    return mp


def main():
    configure_stdout_utf8()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, type=str)
    parser.add_argument("--test_data", default="valid", choices=["valid", "test"])
    parser.add_argument("--candidates_file", required=True, type=str)
    parser.add_argument("--graph_reasoning_type", default="TiRGN", type=str)
    parser.add_argument("--rule_weight", default=0.9, type=float)
    parser.add_argument("--trace_file", default="", type=str)
    parser.add_argument("--bat_file_name", type=str, default="bat_file")
    parser.add_argument("--results_root_path", type=str, default="results")
    args = parser.parse_args()

    args.results_root_path = args.results_root_path.strip('"')
    ranked_rules_dir = get_ranked_rules_dir(args.results_root_path, args.bat_file_name, args.dataset)
    candidates_basename = os.path.splitext(os.path.basename(args.candidates_file))[0]

    trace_name = args.trace_file.strip() or _default_trace_name(
        args.graph_reasoning_type, args.rule_weight, args.candidates_file
    )
    trace_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, trace_name))
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"trace jsonl not found: {trace_path}")

    # Category counts
    n = 0
    no_cand = 0
    no_hit = 0
    hit_top1 = 0
    hit_not_top1 = 0
    hit_not_top1_ranks: List[int] = []

    # Optional fusion diagnostics (top1 correctness change)
    harm = improve = same = 0

    dataset_dir = maybe_windows_long_path(os.path.join("datasets", args.dataset))
    grapher = Grapher(dataset_dir)
    split_np = grapher.test_idx if args.test_data == "test" else grapher.valid_idx
    other_answers_map = _build_other_answers_map(split_np)
    baseline = _load_graph_baseline(args.dataset, dataset_dir, args.graph_reasoning_type, args.test_data)
    if baseline is not None:
        score_numpy, test_index = baseline
    else:
        score_numpy = test_index = None

    iterator = iter_jsonl(trace_path)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Reading trace", unit="lines", mininterval=0.5)
    for row in iterator:
        n += 1
        cand_size = int(row.get("cand_size", 0))
        pred_top1 = int(row.get("pred_top1", -1))
        gt_tail = int(row.get("gt_tail", -1))
        gt_rank_in_cands = row.get("gt_rank_in_cands", None)
        query = row.get("query") or []

        if cand_size == 0:
            no_cand += 1
        else:
            if pred_top1 == gt_tail:
                hit_top1 += 1
            elif gt_rank_in_cands is None:
                no_hit += 1
            else:
                hit_not_top1 += 1
                try:
                    hit_not_top1_ranks.append(int(gt_rank_in_cands))
                except Exception:
                    pass

        # fusion (top1-only) harm/improve/same vs graph-only baseline
        if score_numpy is not None and test_index is not None and len(query) == 4:
            key = tuple(int(x) for x in query)
            pos = test_index.get(key)
            if pos is not None:
                regcn_score = np.array(score_numpy[pos], dtype=np.float32, copy=True)
                s = int(query[0])
                r = int(query[1])
                t = int(query[3])
                objs = other_answers_map.get((s, r, t), [])
                for o in objs:
                    if int(o) != gt_tail:
                        regcn_score[int(o)] = -np.inf
                graph_top1 = int(np.argmax(regcn_score))

                graph_correct = (graph_top1 == gt_tail)
                fused_correct = (pred_top1 == gt_tail)
                if graph_correct and (not fused_correct):
                    harm += 1
                elif (not graph_correct) and fused_correct:
                    improve += 1
                else:
                    same += 1
            else:
                same += 1

    hit_not_top1_ranks_sorted = sorted(hit_not_top1_ranks)
    rank_stats = {
        "count": int(len(hit_not_top1_ranks_sorted)),
        "min": int(hit_not_top1_ranks_sorted[0]) if hit_not_top1_ranks_sorted else None,
        "p50": percentile_from_sorted([float(x) for x in hit_not_top1_ranks_sorted], 50),
        "p90": percentile_from_sorted([float(x) for x in hit_not_top1_ranks_sorted], 90),
        "p95": percentile_from_sorted([float(x) for x in hit_not_top1_ranks_sorted], 95),
        "p99": percentile_from_sorted([float(x) for x in hit_not_top1_ranks_sorted], 99),
        "max": int(hit_not_top1_ranks_sorted[-1]) if hit_not_top1_ranks_sorted else None,
        "mean": float(sum(hit_not_top1_ranks_sorted) / len(hit_not_top1_ranks_sorted)) if hit_not_top1_ranks_sorted else None,
    }

    denom = float(n) if n else 1.0
    payload = {
        "meta": {
            "dataset": args.dataset,
            "split": args.test_data,
            "graph_reasoning_type": args.graph_reasoning_type,
            "rule_weight": float(args.rule_weight),
            "candidates_file": os.path.basename(args.candidates_file),
            "trace_file": os.path.basename(trace_path),
        },
        "N": int(n),
        "NoCand": float(no_cand / denom),
        "NoHit": float(no_hit / denom),
        "HitTop1": float(hit_top1 / denom),
        "HitNotTop1": float(hit_not_top1 / denom),
        "HitNotTop1_rank_stats": rank_stats,
        "harm_rate": float(harm / denom) if score_numpy is not None else 0.0,
        "improve_rate": float(improve / denom) if score_numpy is not None else 0.0,
        "same_rate": float(same / denom) if score_numpy is not None else 0.0,
    }

    run_id = candidates_run_id(args.candidates_file)
    w_str = fmt_float_for_name(args.rule_weight, decimals=2)
    out_base = f"breakdown_{run_id}_{args.graph_reasoning_type}_w{w_str}"
    out_json = maybe_windows_long_path(os.path.join(ranked_rules_dir, out_base + ".json"))
    out_txt = maybe_windows_long_path(os.path.join(ranked_rules_dir, out_base + ".txt"))

    safe_dump_json(out_json, payload, indent=2)

    txt_lines = [
        f"dataset={args.dataset} split={args.test_data}",
        f"graph={args.graph_reasoning_type} rule_weight={args.rule_weight}",
        f"candidates_file={os.path.basename(args.candidates_file)}",
        f"trace_file={os.path.basename(trace_path)}",
        "",
        f"N={payload['N']}",
        f"NoCand={payload['NoCand']:.6f}",
        f"NoHit={payload['NoHit']:.6f}",
        f"HitTop1={payload['HitTop1']:.6f}",
        f"HitNotTop1={payload['HitNotTop1']:.6f}",
        "",
        f"HitNotTop1_rank_stats={rank_stats}",
        "",
        f"harm_rate={payload['harm_rate']:.6f}",
        f"improve_rate={payload['improve_rate']:.6f}",
        f"same_rate={payload['same_rate']:.6f}",
        "",
    ]
    safe_write_text(out_txt, "\n".join(txt_lines))

    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_txt}")


if __name__ == "__main__":
    main()
