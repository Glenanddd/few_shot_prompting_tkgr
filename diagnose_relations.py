import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

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
    safe_write_csv,
    safe_write_text,
    topk_counter,
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


def _format_top_entities(top_list: List[Dict[str, int]]) -> str:
    return ";".join([f"{d['eid']}:{d['cnt']}" for d in top_list])


def main():
    configure_stdout_utf8()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, type=str)
    parser.add_argument("--test_data", default="valid", choices=["valid", "test"])
    parser.add_argument("--candidates_file", required=True, type=str)
    parser.add_argument("--graph_reasoning_type", default="TiRGN", type=str)
    parser.add_argument("--rule_weight", default=0.9, type=float)
    parser.add_argument("--trace_file", default="", type=str)
    parser.add_argument("--min_n", default=50, type=int)
    parser.add_argument("--top_relations", default=50, type=int)
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

    dataset_dir = maybe_windows_long_path(os.path.join("datasets", args.dataset))
    data = Grapher(dataset_dir)

    # per relation accumulators
    n_by_rel = Counter()
    nohit_by_rel = Counter()     # gt not in candidates (including NoCand)
    hit_by_rel = Counter()       # gt in candidates
    top1_hit_by_rel = Counter()  # pred_top1 == gt
    wrong_top1_by_rel = Counter()
    cand_sizes_by_rel: Dict[int, List[int]] = defaultdict(list)
    ans_rank_not_top1_by_rel: Dict[int, List[int]] = defaultdict(list)
    wrong_top1_entities_by_rel: Dict[int, Counter] = defaultdict(Counter)

    iterator = iter_jsonl(trace_path)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Reading trace", unit="lines", mininterval=0.5)
    for row in iterator:
        rel = int(row.get("relation", -1))
        n_by_rel[rel] += 1

        cand_size = int(row.get("cand_size", 0))
        cand_sizes_by_rel[rel].append(cand_size)

        gt_rank_in_cands = row.get("gt_rank_in_cands", None)
        pred_top1 = int(row.get("pred_top1", -1))
        gt_tail = int(row.get("gt_tail", -1))

        if gt_rank_in_cands is None:
            nohit_by_rel[rel] += 1
        else:
            hit_by_rel[rel] += 1

        if pred_top1 == gt_tail:
            top1_hit_by_rel[rel] += 1
        else:
            wrong_top1_by_rel[rel] += 1
            wrong_top1_entities_by_rel[rel][pred_top1] += 1
            if gt_rank_in_cands is not None:
                try:
                    ans_rank_not_top1_by_rel[rel].append(int(gt_rank_in_cands))
                except Exception:
                    pass

    rows_json: List[Dict[str, Any]] = []
    rows_csv: List[Dict[str, Any]] = []

    for rel, n in n_by_rel.items():
        if int(n) < int(args.min_n):
            continue

        hit_n = int(hit_by_rel.get(rel, 0))
        top1_hit_n = int(top1_hit_by_rel.get(rel, 0))
        wrong_n = int(wrong_top1_by_rel.get(rel, 0))
        nohit_n = int(nohit_by_rel.get(rel, 0))

        nohit_rate = float(nohit_n / n) if n else 0.0
        hit_rate = float(hit_n / n) if n else 0.0
        cond_top1 = float(top1_hit_n / hit_n) if hit_n else 0.0
        wrong_rate = float(wrong_n / n) if n else 0.0

        cand_sizes = sorted([float(x) for x in cand_sizes_by_rel.get(rel, [])])
        cand_p95 = percentile_from_sorted(cand_sizes, 95)

        ans_ranks = sorted([float(x) for x in ans_rank_not_top1_by_rel.get(rel, [])])
        ans_rank_p95 = percentile_from_sorted(ans_ranks, 95)

        top_wrong_entities = topk_counter(
            {int(e): int(c) for e, c in wrong_top1_entities_by_rel.get(rel, Counter()).items()},
            10,
        )

        rel_name = data.id2relation.get(int(rel), str(rel))

        worst_score = float(nohit_rate + wrong_rate)
        item = {
            "rel_id": int(rel),
            "rel_name": rel_name,
            "n_test": int(n),
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
        rows_json.append(item)

        rows_csv.append(
            {
                "rel_id": item["rel_id"],
                "rel_name": item["rel_name"],
                "n_test": item["n_test"],
                "NoHit_rate_test": item["NoHit_rate_test"],
                "Hit_rate_test": item["Hit_rate_test"],
                "CondTop1_given_Hit": item["CondTop1_given_Hit"],
                "WrongTop1_total": item["WrongTop1_total"],
                "WrongTop1_rate": item["WrongTop1_rate"],
                "Cand_p95": item["Cand_p95"],
                "AnsRankNotTop1_p95": item["AnsRankNotTop1_p95"],
                "TopWrongTop1Entities": _format_top_entities(top_wrong_entities),
                "worst_score": worst_score,
            }
        )

    rows_json.sort(key=lambda x: (-float(x.get("_worst_score", 0.0)), -int(x.get("n_test", 0)), int(x.get("rel_id", 0))))

    run_id = candidates_run_id(args.candidates_file)
    w_str = fmt_float_for_name(args.rule_weight, decimals=2)
    out_base = f"rel_diag_{run_id}_{args.graph_reasoning_type}_w{w_str}"
    out_json = maybe_windows_long_path(os.path.join(ranked_rules_dir, out_base + ".json"))
    out_csv = maybe_windows_long_path(os.path.join(ranked_rules_dir, out_base + ".csv"))
    out_txt = maybe_windows_long_path(os.path.join(ranked_rules_dir, out_base + ".txt"))

    safe_dump_json(
        out_json,
        {
            "meta": {
                "dataset": args.dataset,
                "split": args.test_data,
                "graph_reasoning_type": args.graph_reasoning_type,
                "rule_weight": float(args.rule_weight),
                "candidates_file": os.path.basename(args.candidates_file),
                "trace_file": os.path.basename(trace_path),
                "min_n": int(args.min_n),
            },
            "relations": [{k: v for k, v in r.items() if k != "_worst_score"} for r in rows_json],
        },
        indent=2,
    )

    fieldnames = [
        "rel_id",
        "rel_name",
        "n_test",
        "NoHit_rate_test",
        "Hit_rate_test",
        "CondTop1_given_Hit",
        "WrongTop1_total",
        "WrongTop1_rate",
        "Cand_p95",
        "AnsRankNotTop1_p95",
        "TopWrongTop1Entities",
        "worst_score",
    ]
    safe_write_csv(out_csv, rows_csv, fieldnames)

    top_show = int(args.top_relations)
    txt_lines = [
        f"dataset={args.dataset} split={args.test_data}",
        f"graph={args.graph_reasoning_type} rule_weight={args.rule_weight}",
        f"candidates_file={os.path.basename(args.candidates_file)}",
        f"trace_file={os.path.basename(trace_path)}",
        f"min_n={args.min_n}",
        "",
        f"Top worst relations (top {top_show}):",
    ]
    for r in rows_json[:top_show]:
        txt_lines.append(
            f"rel_id={r['rel_id']} n={r['n_test']} NoHit={r['NoHit_rate_test']:.3f} "
            f"WrongTop1={r['WrongTop1_rate']:.3f} CondTop1|Hit={r['CondTop1_given_Hit']:.3f} "
            f"Cand_p95={r['Cand_p95']} AnsRankNotTop1_p95={r['AnsRankNotTop1_p95']}"
        )
    txt_lines.append("")
    safe_write_text(out_txt, "\n".join(txt_lines))

    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_txt}")


if __name__ == "__main__":
    main()
