import argparse
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List

from grapher import Grapher
from utils_windows_long_path import maybe_windows_long_path
from vlrg_utils import (
    candidates_run_id,
    configure_stdout_utf8,
    fmt_float_for_name,
    get_ranked_rules_dir,
    iter_jsonl,
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


def _format_entity_list(top_list: List[Dict[str, int]]) -> str:
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
    parser.add_argument("--pop_top", default=15, type=int)
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

    # From trace: count wrong top1 entities per relation.
    n_by_rel = Counter()
    wrong_top1_entities_by_rel: Dict[int, Counter] = defaultdict(Counter)
    iterator = iter_jsonl(trace_path)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Reading trace", unit="lines", mininterval=0.5)
    for row in iterator:
        rel = int(row.get("relation", -1))
        n_by_rel[rel] += 1
        pred = int(row.get("pred_top1", -1))
        gt = int(row.get("gt_tail", -1))
        if pred != gt:
            wrong_top1_entities_by_rel[rel][pred] += 1

    # Train tail popularity per relation.
    tail_freq_by_rel: Dict[int, Counter] = defaultdict(Counter)
    for row in data.train_idx:
        r = int(row[1])
        tail = int(row[2])
        tail_freq_by_rel[r][tail] += 1

    out_rows_json: List[Dict[str, Any]] = []
    out_rows_csv: List[Dict[str, Any]] = []

    pop_top = int(args.pop_top)
    for rel, n in n_by_rel.items():
        if int(n) < int(args.min_n):
            continue

        wrong_counter = wrong_top1_entities_by_rel.get(rel, Counter())
        top_wrong = topk_counter({int(e): int(c) for e, c in wrong_counter.items()}, 10)
        top_wrong_eids = [int(x["eid"]) for x in top_wrong]

        train_counter = tail_freq_by_rel.get(rel, Counter())
        top_train_items = train_counter.most_common(pop_top)
        top_train = [{"eid": int(e), "cnt": int(c)} for e, c in top_train_items]
        top_train_eids = [int(x["eid"]) for x in top_train]

        overlap_set = set(top_train_eids)
        overlap = [eid for eid in top_wrong_eids if eid in overlap_set]
        overlap_num = int(len(overlap))

        avg_rank = None
        if overlap:
            full_sorted = [eid for eid, _ in train_counter.most_common()]
            idx_map = {int(eid): int(i + 1) for i, eid in enumerate(full_sorted)}
            ranks = [idx_map[eid] for eid in overlap if eid in idx_map]
            if ranks:
                avg_rank = float(sum(ranks) / len(ranks))

        rel_name = data.id2relation.get(int(rel), str(rel))

        item = {
            "rel_id": int(rel),
            "rel_name": rel_name,
            "n_test": int(n),
            "WrongTop1_top@10": top_wrong,
            f"TrainTail_top@{pop_top}": top_train,
            "overlap_num": overlap_num,
            "overlapped_wrong_avg_train_rank": avg_rank,
        }
        out_rows_json.append(item)

        out_rows_csv.append(
            {
                "rel_id": item["rel_id"],
                "rel_name": item["rel_name"],
                "n_test": item["n_test"],
                "overlap_num": item["overlap_num"],
                "overlapped_wrong_avg_train_rank": item["overlapped_wrong_avg_train_rank"],
                "top_wrong": _format_entity_list(top_wrong),
                "top_train": _format_entity_list(top_train),
            }
        )

    out_rows_json.sort(key=lambda x: (-int(x.get("overlap_num", 0)), -int(x.get("n_test", 0)), int(x.get("rel_id", 0))))

    run_id = candidates_run_id(args.candidates_file)
    w_str = fmt_float_for_name(args.rule_weight, decimals=2)
    out_base = f"popbias_{run_id}_{args.graph_reasoning_type}_w{w_str}"
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
                "pop_top": pop_top,
            },
            "relations": out_rows_json,
        },
        indent=2,
    )

    fieldnames = [
        "rel_id",
        "rel_name",
        "n_test",
        "overlap_num",
        "overlapped_wrong_avg_train_rank",
        "top_wrong",
        "top_train",
    ]
    safe_write_csv(out_csv, out_rows_csv, fieldnames)

    txt_lines = [
        f"dataset={args.dataset} split={args.test_data}",
        f"graph={args.graph_reasoning_type} rule_weight={args.rule_weight}",
        f"candidates_file={os.path.basename(args.candidates_file)}",
        f"trace_file={os.path.basename(trace_path)}",
        f"min_n={args.min_n} pop_top={pop_top}",
        "",
        "Top relations with popularity overlap (sorted by overlap_num):",
    ]
    for r in out_rows_json[: min(30, len(out_rows_json))]:
        txt_lines.append(
            f"rel_id={r['rel_id']} overlap_num={r['overlap_num']} n={r['n_test']} avg_rank={r['overlapped_wrong_avg_train_rank']}"
        )
    txt_lines.append("")
    safe_write_text(out_txt, "\n".join(txt_lines))

    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_txt}")


if __name__ == "__main__":
    main()
