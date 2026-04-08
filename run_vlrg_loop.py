import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from grapher import Grapher
from utils_windows_long_path import maybe_windows_long_path, safe_open
from vlrg_utils import (
    cfg_id_from_config,
    configure_stdout_utf8,
    fmt_float_for_name,
    get_ranked_rules_dir,
    load_json,
    round_tag,
    run_id as make_run_id,
    safe_dump_json,
    safe_write_csv,
    safe_write_text,
)


def _run(cmd: List[str]) -> float:
    # Keep console output ASCII-safe.
    t0 = time.time()
    print("[Run]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return float(time.time() - t0)


def _fmt_hhmmss(seconds: float) -> str:
    try:
        s = int(max(0.0, float(seconds)))
    except Exception:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


def _print_eta(
    *,
    start_time: float,
    total_iters: int,
    done_iters: int,
    last_iters_sec: List[float],
    est_final_test_sec: Optional[float],
) -> None:
    elapsed = time.time() - float(start_time)

    window = last_iters_sec[-min(5, len(last_iters_sec)) :] if last_iters_sec else []
    if window:
        avg_iter = float(sum(window) / len(window))
    elif done_iters > 0:
        avg_iter = float(elapsed / done_iters)
    else:
        print(f"[ETA] elapsed={_fmt_hhmmss(elapsed)} (warming up)")
        return
    remaining_iters = max(0, int(total_iters) - int(done_iters))
    eta_loop = float(remaining_iters) * float(avg_iter)

    eta_total = eta_loop
    if est_final_test_sec is not None:
        try:
            eta_total += float(est_final_test_sec)
        except Exception:
            pass

    finish_at = datetime.now() + timedelta(seconds=float(eta_total))
    fin_str = finish_at.strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        f"[ETA] iters {done_iters}/{total_iters} "
        f"elapsed={_fmt_hhmmss(elapsed)} "
        f"avg_iter={_fmt_hhmmss(avg_iter)} "
        f"remaining={_fmt_hhmmss(eta_loop)}"
    )
    if est_final_test_sec is not None:
        msg += f" +final_test~{_fmt_hhmmss(est_final_test_sec)}"
    msg += f" finish_at={fin_str}"
    print(msg)


def _safe_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        shutil.copy2(maybe_windows_long_path(src), maybe_windows_long_path(dst))
    except PermissionError as e:
        raise PermissionError(
            f"PermissionError copying file:\n  src={src}\n  dst={dst}\n"
            "This can happen if the destination is opened in Excel or another program."
        ) from e


def _safe_move(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.replace(maybe_windows_long_path(src), maybe_windows_long_path(dst))
    except PermissionError as e:
        raise PermissionError(
            f"PermissionError moving file:\n  src={src}\n  dst={dst}\n"
            "This can happen if the destination is opened in Excel or another program."
        ) from e


def _safe_remove(path: str) -> None:
    try:
        os.remove(maybe_windows_long_path(path))
    except FileNotFoundError:
        return
    except PermissionError as e:
        raise PermissionError(
            f"PermissionError deleting file: {path}\n"
            "This can happen if the file is opened in Excel or another program."
        ) from e
    except Exception:
        return


def _safe_dump_json_compact(path: str, obj: Any) -> None:
    try:
        with safe_open(maybe_windows_long_path(path), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except PermissionError as e:
        raise PermissionError(
            f"PermissionError writing JSON: {path}\n"
            "This can happen if the file is opened in Excel or another program."
        ) from e


def _file_confidence(round_k: int) -> str:
    return f"confidence_{round_tag(round_k)}.json"


def _file_confidence_concrete(round_k: int) -> str:
    return f"confidence_concrete_{round_tag(round_k)}.json"


def _file_candidates(run_id: str, *, partial: bool) -> str:
    prefix = "cands_partial" if partial else "cands"
    return f"{prefix}_{run_id}.json"


def _file_trace(run_id: str, graph: str, w_str: str, *, partial: bool) -> str:
    prefix = "trace_partial" if partial else "trace"
    return f"{prefix}_{run_id}_{graph}_w{w_str}.jsonl"


def _file_metrics(run_id: str, graph: str, w_str: str, ext: str) -> str:
    return f"metrics_{run_id}_{graph}_w{w_str}.{ext}"


def _file_breakdown(run_id: str, graph: str, w_str: str, ext: str) -> str:
    return f"breakdown_{run_id}_{graph}_w{w_str}.{ext}"


def _file_rel_diag(run_id: str, graph: str, w_str: str, ext: str) -> str:
    return f"rel_diag_{run_id}_{graph}_w{w_str}.{ext}"


def _file_popbias(run_id: str, graph: str, w_str: str, ext: str) -> str:
    return f"popbias_{run_id}_{graph}_w{w_str}.{ext}"


def _file_evidence(run_id: str, graph: str, w_str: str) -> str:
    return f"evidence_for_llm_{run_id}_{graph}_w{w_str}.json"


def _file_qids(run_id: str) -> str:
    return f"qids_{run_id}.json"


def _file_patch(round_to: int) -> str:
    return f"patch_{round_tag(round_to)}.json"


def _count_non_empty_lines(path: str) -> int:
    n = 0
    with safe_open(maybe_windows_long_path(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if (line or "").strip():
                n += 1
    return n


def _collect_changed_rule_ids(patch_path: str) -> Set[str]:
    patch = load_json(patch_path, default=None)
    if not isinstance(patch, dict):
        return set()
    actions = patch.get("actions") if isinstance(patch.get("actions"), dict) else {}

    changed: Set[str] = set()
    for x in (actions.get("delete") or []):
        if x:
            changed.add(str(x))
    for k in (actions.get("downweight") or {}).keys():
        if k:
            changed.add(str(k))
    for k in (actions.get("promote") or {}).keys():
        if k:
            changed.add(str(k))
    return changed


def _collect_affected_relations(conf_concrete_path: str, changed_rule_ids: Set[str]) -> Set[int]:
    payload = load_json(conf_concrete_path, default={}) or {}
    if not isinstance(payload, dict) or not changed_rule_ids:
        return set()

    rels: Set[int] = set()
    for rel_key, rules in payload.items():
        if not isinstance(rules, list):
            continue
        hit = False
        for r in rules:
            if not isinstance(r, dict):
                continue
            rid = r.get("rule_id")
            if rid and str(rid) in changed_rule_ids:
                hit = True
                break
        if hit:
            try:
                rels.add(int(rel_key))
            except Exception:
                pass
    return rels


def _select_qids_from_trace(trace_path: str, affected_relations: Set[int]) -> List[int]:
    if not affected_relations:
        return []
    qids: Set[int] = set()
    with safe_open(maybe_windows_long_path(trace_path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            try:
                rel = int(row.get("relation", -1))
                if rel not in affected_relations:
                    continue
                qid = int(row.get("qid", -1))
            except Exception:
                continue
            if qid >= 0:
                qids.add(qid)
    return sorted(qids)


def _select_qids_from_split_np(split_np, affected_relations: Set[int]) -> List[int]:
    """
    No-trace incremental reasoning:
    select qids whose relation id is in affected_relations.
    """
    if not affected_relations:
        return []
    qids: List[int] = []
    try:
        for i, row in enumerate(split_np):
            try:
                rel = int(row[1])
            except Exception:
                continue
            if rel in affected_relations:
                qids.append(int(i))
    except Exception:
        return []
    return qids


def _merge_candidates(prev_path: str, partial_path: str, out_path: str, total_queries: int) -> None:
    prev = load_json(prev_path, default={}) or {}
    part = load_json(partial_path, default={}) or {}
    if not isinstance(prev, dict) or not isinstance(part, dict):
        raise ValueError("Invalid candidates JSON: expected dict objects.")

    merged = dict(prev)
    merged.update(part)

    ordered: Dict[str, Any] = {}
    for i in range(int(total_queries)):
        k = str(int(i))
        ordered[k] = merged.get(k, {}) if merged.get(k, None) is not None else {}
    _safe_dump_json_compact(out_path, ordered)


def _merge_trace(prev_path: str, partial_path: str, out_path: str) -> None:
    part_map: Dict[int, str] = {}
    with safe_open(maybe_windows_long_path(partial_path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                row = json.loads(s)
                qid = int(row.get("qid", -1))
            except Exception:
                continue
            if qid >= 0:
                part_map[qid] = s

    with safe_open(maybe_windows_long_path(out_path), "w", encoding="utf-8", errors="replace") as fout:
        with safe_open(maybe_windows_long_path(prev_path), "r", encoding="utf-8", errors="replace") as fin:
            for line in fin:
                s = (line or "").strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                    qid = int(row.get("qid", -1))
                except Exception:
                    fout.write(line)
                    continue
                if qid in part_map:
                    fout.write(part_map.pop(qid) + "\n")
                else:
                    fout.write(s + "\n")
        if part_map:
            for qid in sorted(part_map.keys()):
                fout.write(part_map[qid] + "\n")


def _build_reasoning_cmd(
    py: str,
    *,
    dataset: str,
    split: str,
    query_dir: str,
    rules_confidence_file: str,
    bgkg: str,
    gpu: int,
    top_k: int,
    num_processes: int,
    window: int,
    rule_lengths: List[int],
    evaluation_type: str,
    confidence_type: str,
    min_conf: float,
    lmbda: float,
    weight_0: float,
    weight: float,
    coor_weight: float,
    win_start: int,
    is_sorted: str,
    is_relax_time: str,
    is_sampled: str,
    is_rule_priority: str,
    trace_rules: str,
    trace_top_rules: int,
    graph_reasoning_type: str,
    rule_weight: float,
    candidates_file_name: str,
    trace_file_name: Optional[str],
    qids_file: Optional[str],
    bat_file_name: str,
    results_root_path: str,
) -> List[str]:
    cmd: List[str] = [
        py,
        "reasoning.py",
        "-d",
        dataset,
        "--test_data",
        split,
        "--query_dir",
        str(query_dir),
        "-r",
        rules_confidence_file,
        "--bgkg",
        bgkg,
        "--gpu",
        str(int(gpu)),
        "--top_k",
        str(int(top_k)),
        "--num_processes",
        str(int(num_processes)),
        "--window",
        str(int(window)),
        "--rule_lengths",
        *[str(int(x)) for x in (rule_lengths or [])],
        "--evaluation_type",
        evaluation_type,
        "--confidence_type",
        confidence_type,
        "--min_conf",
        str(float(min_conf)),
        "--lmbda",
        str(float(lmbda)),
        "--weight_0",
        str(float(weight_0)),
        "--weight",
        str(float(weight)),
        "--coor_weight",
        str(float(coor_weight)),
        "--win_start",
        str(int(win_start)),
        "--is_sorted",
        is_sorted,
        "--is_relax_time",
        is_relax_time,
        "--is_sampled",
        is_sampled,
        "--is_rule_priority",
        is_rule_priority,
        "--trace_rules",
        trace_rules,
        "--trace_top_rules",
        str(int(trace_top_rules)),
        "--graph_reasoning_type",
        graph_reasoning_type,
        "--rule_weight",
        str(float(rule_weight)),
        "--candidates_file_name",
        os.path.basename(str(candidates_file_name)),
    ]
    if trace_file_name:
        cmd += ["--trace_file_name", os.path.basename(str(trace_file_name))]
    if qids_file:
        cmd += ["--qids_file", str(qids_file)]
    cmd += ["--bat_file_name", bat_file_name, "--results_root_path", results_root_path]
    return cmd


def _build_evaluate_cmd(
    py: str,
    *,
    dataset: str,
    split: str,
    query_dir: str,
    candidates_file_name: str,
    graph_reasoning_type: str,
    rule_weight: float,
    bat_file_name: str,
    results_root_path: str,
) -> List[str]:
    return [
        py,
        "evaluate.py",
        "-d",
        dataset,
        "--test_data",
        split,
        "--query_dir",
        str(query_dir),
        "--candidates_file_name",
        os.path.basename(str(candidates_file_name)),
        "--graph_reasoning_type",
        graph_reasoning_type,
        "--rule_weight",
        str(float(rule_weight)),
        "--bat_file_name",
        bat_file_name,
        "--results_root_path",
        results_root_path,
    ]


def main() -> None:
    configure_stdout_utf8()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", type=str, choices=["full", "test_only"])
    parser.add_argument("--dataset", "-d", required=True, type=str)
    parser.add_argument("--K", default=10, type=int)
    # Rank-rule bgkg: must be train to avoid leakage.
    parser.add_argument("--bgkg", default="train", type=str, choices=["train"])
    parser.add_argument("--graph_reasoning_type", default="TiRGN", type=str)
    parser.add_argument("--rule_weight", default=0.9, type=float)
    parser.add_argument("--test_data", default="valid", choices=["valid"])
    parser.add_argument("--query_dir", default="both", choices=["both", "forward"])
    parser.add_argument("--final_test", default="No", type=str, choices=["Yes", "No"])
    parser.add_argument("--test_all_rounds", default="No", type=str, choices=["Yes", "No"])
    parser.add_argument("--llm_model", default="gpt-5-nano", type=str)
    parser.add_argument("--llm_dry_run", action="store_true")
    parser.add_argument("--incremental_reasoning", default="Yes", type=str, choices=["Yes", "No"])
    parser.add_argument("--trace_rules", default="Yes", type=str, choices=["Yes", "No"])
    parser.add_argument("--trace_top_rules", default=5, type=int)
    parser.add_argument("--allow_delete", default="Yes", type=str, choices=["Yes", "No"])
    parser.add_argument("--allow_promote", default="Yes", type=str, choices=["Yes", "No"])

    # Pass-through (consistent with other scripts)
    parser.add_argument("--bat_file_name", type=str, default="bat_file")
    parser.add_argument("--results_root_path", type=str, default="results")

    # Reasoning knobs
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--window", type=int, default=-1)
    parser.add_argument("--rule_lengths", type=int, nargs="+", default=[1])
    parser.add_argument("--evaluation_type", type=str, default="origin", choices=["origin", "transformer", "end"])
    parser.add_argument("--confidence_type", type=str, default="Common", choices=["Common", "LLM", "And", "Or"])
    parser.add_argument("--min_conf", type=float, default=0.01)
    parser.add_argument("--lmbda", type=float, default=0.1)
    parser.add_argument("--weight_0", type=float, default=0.5)
    parser.add_argument("--weight", type=float, default=0.0)
    parser.add_argument("--coor_weight", type=float, default=0.0)
    parser.add_argument("--win_start", type=int, default=0)
    parser.add_argument("--is_sorted", type=str, default="Yes", choices=["Yes", "No"])
    parser.add_argument("--is_relax_time", type=str, default="no", choices=["yes", "no"])
    parser.add_argument("--is_sampled", type=str, default="no", choices=["yes", "no"])
    parser.add_argument("--is_rule_priority", type=str, default="no", choices=["yes", "no"])

    args = parser.parse_args()

    args.results_root_path = args.results_root_path.strip('"')
    ranked_rules_dir = get_ranked_rules_dir(args.results_root_path, args.bat_file_name, args.dataset)
    os.makedirs(ranked_rules_dir, exist_ok=True)

    py = sys.executable
    graph = str(args.graph_reasoning_type)
    w = float(args.rule_weight)
    w_str = fmt_float_for_name(w, decimals=2)

    # cfg_id for short, stable filenames; store full config in cfg_*.json
    cfg = {
        "dataset": args.dataset,
        "graph_reasoning_type": graph,
        "rule_weight": float(w),
        "rule_lengths": [int(x) for x in (args.rule_lengths or [])],
        "window": int(args.window),
        "evaluation_type": str(args.evaluation_type),
        "confidence_type": str(args.confidence_type),
        "min_conf": float(args.min_conf),
        "lmbda": float(args.lmbda),
        "weight_0": float(args.weight_0),
        "weight": float(args.weight),
        "coor_weight": float(args.coor_weight),
        "win_start": int(args.win_start),
        "top_k": int(args.top_k),
        "is_sorted": str(args.is_sorted),
        "is_relax_time": str(args.is_relax_time),
        "is_sampled": str(args.is_sampled),
        "is_rule_priority": str(args.is_rule_priority),
        "num_processes": int(args.num_processes),
    }
    if str(args.query_dir).lower().strip() != "both":
        cfg["query_dir"] = str(args.query_dir).lower().strip()
    cfg_id = cfg_id_from_config(cfg, length=8)
    cfg_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, f"cfg_{cfg_id}.json"))
    if args.mode == "test_only":
        # Prefer reusing an existing cfg_id for consistent filenames.
        # For non-default query_dir, try to match query_dir; otherwise keep the computed cfg_id.
        try:
            qd = str(args.query_dir).lower().strip()
            chosen_id = None
            if not os.path.exists(cfg_path):
                for name in os.listdir(ranked_rules_dir):
                    if not (name.startswith("cfg_") and name.endswith(".json")):
                        continue
                    cand_id = name[len("cfg_") : -len(".json")]
                    cand_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, name))
                    payload = load_json(cand_path, default={}) or {}
                    cfg_obj = payload.get("cfg") if isinstance(payload, dict) else {}
                    qd_in = str(cfg_obj.get("query_dir", "both")).lower().strip() if isinstance(cfg_obj, dict) else "both"
                    if qd_in == qd:
                        chosen_id = cand_id
                        break
                    if qd == "both" and chosen_id is None:
                        # Backward compatible fallback: reuse the first cfg_*.json.
                        chosen_id = cand_id
            if chosen_id:
                cfg_id = chosen_id
                cfg_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, f"cfg_{cfg_id}.json"))
        except Exception:
            pass
    if args.mode != "test_only" and not os.path.exists(cfg_path):
        safe_dump_json(cfg_path, {"cfg_id": cfg_id, "cfg": cfg}, indent=2)

    start_time = time.time()

    def run_test_round(round_k: int) -> Dict[str, Any]:
        rid_test = make_run_id(round_k, "test", cfg_id)
        cands_test = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_candidates(rid_test, partial=False)))
        metrics_json = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_metrics(rid_test, graph, w_str, "json")))
        metrics_csv = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_metrics(rid_test, graph, w_str, "csv")))

        # Allow resume: skip re-run if metrics exist.
        if not os.path.exists(metrics_json):
            if not os.path.exists(cands_test):
                _run(
                    _build_reasoning_cmd(
                        py,
                        dataset=args.dataset,
                        split="test",
                        query_dir=str(args.query_dir),
                        rules_confidence_file=_file_confidence(round_k),
                        bgkg="all",
                        gpu=args.gpu,
                        top_k=args.top_k,
                        num_processes=args.num_processes,
                        window=args.window,
                        rule_lengths=args.rule_lengths,
                        evaluation_type=args.evaluation_type,
                        confidence_type=args.confidence_type,
                        min_conf=args.min_conf,
                        lmbda=args.lmbda,
                        weight_0=args.weight_0,
                        weight=args.weight,
                        coor_weight=args.coor_weight,
                        win_start=args.win_start,
                        is_sorted=args.is_sorted,
                        is_relax_time=args.is_relax_time,
                        is_sampled=args.is_sampled,
                        is_rule_priority=args.is_rule_priority,
                        trace_rules="No",
                        trace_top_rules=5,
                        graph_reasoning_type=graph,
                        rule_weight=w,
                        candidates_file_name=cands_test,
                        trace_file_name=None,
                        qids_file=None,
                        bat_file_name=args.bat_file_name,
                        results_root_path=args.results_root_path,
                    )
                )
            _run(
                _build_evaluate_cmd(
                    py,
                    dataset=args.dataset,
                    split="test",
                    query_dir=str(args.query_dir),
                    candidates_file_name=cands_test,
                    graph_reasoning_type=graph,
                    rule_weight=w,
                    bat_file_name=args.bat_file_name,
                    results_root_path=args.results_root_path,
                )
            )

        m_payload = load_json(metrics_json, default={}) or {}
        picked = m_payload.get("metrics", {}) if isinstance(m_payload.get("metrics"), dict) else {}
        return {
            "round": int(round_k),
            "split": "test",
            "cfg_id": cfg_id,
            "run_id": rid_test,
            "confidence_file": _file_confidence(round_k),
            "candidates_file": os.path.basename(cands_test),
            "metrics_json": os.path.basename(metrics_json),
            "metrics_csv": os.path.basename(metrics_csv) if os.path.exists(metrics_csv) else "",
            "mrr": float(picked.get("mrr", 0.0)) if picked else 0.0,
            "hits1": float(picked.get("hits1", 0.0)) if picked else 0.0,
            "hits3": float(picked.get("hits3", 0.0)) if picked else 0.0,
            "hits10": float(picked.get("hits10", 0.0)) if picked else 0.0,
        }

    if args.mode == "test_only":
        rounds: List[int] = []
        for name in os.listdir(ranked_rules_dir):
            if not (name.startswith("confidence_round_") and name.endswith(".json")):
                continue
            s = name[len("confidence_round_") : -len(".json")]
            if s.isdigit():
                rounds.append(int(s))
        rounds = sorted(set(rounds))
        if not rounds:
            raise FileNotFoundError(f"No confidence_round_*.json found in ranked_rules_dir: {ranked_rules_dir}")

        best_round = max(rounds)
        best_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, "best_round.txt"))
        if os.path.exists(best_path):
            try:
                with safe_open(best_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        s = (line or "").strip()
                        if s.startswith("best_round="):
                            best_round = int(s.split("=", 1)[1].strip())
                            break
            except Exception:
                pass

        test_rows: List[Dict[str, Any]] = []
        if args.test_all_rounds == "Yes":
            for rr in rounds:
                test_rows.append(run_test_round(rr))
            safe_write_csv(
                maybe_windows_long_path(os.path.join(ranked_rules_dir, "test_summary.csv")),
                test_rows,
                [
                    "round",
                    "split",
                    "cfg_id",
                    "run_id",
                    "mrr",
                    "hits1",
                    "hits3",
                    "hits10",
                    "confidence_file",
                    "candidates_file",
                    "metrics_json",
                    "metrics_csv",
                ],
            )

        if args.final_test == "Yes":
            best_row = None
            for tr in test_rows:
                if int(tr.get("round", -1)) == int(best_round):
                    best_row = tr
                    break
            if best_row is None:
                best_row = run_test_round(best_round)

            best_metrics_json = maybe_windows_long_path(os.path.join(ranked_rules_dir, str(best_row.get("metrics_json"))))
            best_metrics_csv = maybe_windows_long_path(os.path.join(ranked_rules_dir, str(best_row.get("metrics_csv"))))
            m_payload = load_json(best_metrics_json, default={}) or {}
            safe_dump_json(maybe_windows_long_path(os.path.join(ranked_rules_dir, "final_test_metrics.json")), m_payload, indent=2)
            if best_metrics_csv and os.path.exists(best_metrics_csv):
                _safe_copy(best_metrics_csv, maybe_windows_long_path(os.path.join(ranked_rules_dir, "final_test_metrics.csv")))

        if args.test_all_rounds != "Yes" and args.final_test != "Yes":
            print("[Done] test_only: nothing to run (set --test_all_rounds Yes and/or --final_test Yes)")
        else:
            if args.test_all_rounds == "Yes":
                print(f"[Done] test_summary => {maybe_windows_long_path(os.path.join(ranked_rules_dir, 'test_summary.csv'))}")
            if args.final_test == "Yes":
                print(f"[Done] final_test_metrics => {maybe_windows_long_path(os.path.join(ranked_rules_dir, 'final_test_metrics.json'))}")
        return

    # ----------------------------
    # Round 0: rank rules (train)
    # ----------------------------
    _run(
        [
            py,
            "rank_rule.py",
            "-d",
            args.dataset,
            "--num_workers",
            "16",
            "--base_seed",
            "1",
            "--stable_merge",
            "Yes",
            "--bgkg",
            "train",
            "--bat_file_name",
            args.bat_file_name,
            "--results_root_path",
            args.results_root_path,
        ]
    )

    conf0_src = maybe_windows_long_path(os.path.join(ranked_rules_dir, "confidence.json"))
    conc0_src = maybe_windows_long_path(os.path.join(ranked_rules_dir, "confidence_concrete.json"))
    conf0 = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_confidence(0)))
    conc0 = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_confidence_concrete(0)))
    if not os.path.exists(conf0_src) or not os.path.exists(conc0_src):
        raise FileNotFoundError("rank_rule.py did not produce confidence.json/confidence_concrete.json in ranked_rules_dir")
    _safe_move(conf0_src, conf0)
    _safe_move(conc0_src, conc0)

    # Lazy-load VALID split for no-trace incremental reasoning (avoid depending on trace file).
    valid_split_np = None
    total_valid_queries = None

    def _ensure_valid_split():
        nonlocal valid_split_np, total_valid_queries
        if valid_split_np is None:
            dataset_dir = maybe_windows_long_path(os.path.join("datasets", args.dataset))
            g = Grapher(dataset_dir)
            valid_split_np = g.valid_idx
            total_valid_queries = int(len(valid_split_np))
        return valid_split_np, int(total_valid_queries or 0)

    # Helper: run one VALID round, optionally incremental by patch
    def run_valid_round(round_k: int, *, patch_path: Optional[str]) -> Dict[str, Any]:
        t_round_start = time.time()

        conf_file = _file_confidence(round_k)
        conc_file = _file_confidence_concrete(round_k)
        conf_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, conf_file))
        conc_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, conc_file))
        if not os.path.exists(conf_path) or not os.path.exists(conc_path):
            raise FileNotFoundError(f"Missing confidence files for round {round_k}: {conf_file} / {conc_file}")

        rid = make_run_id(round_k, "valid", cfg_id)
        cands_full = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_candidates(rid, partial=False)))
        trace_enabled = str(getattr(args, "trace_rules", "Yes")).strip() == "Yes"
        trace_full = (
            maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_trace(rid, graph, w_str, partial=False)))
            if trace_enabled
            else None
        )

        do_incremental = (args.incremental_reasoning == "Yes") and (round_k > 0) and bool(patch_path)
        changed_rule_ids = _collect_changed_rule_ids(str(patch_path)) if patch_path else set()

        time_reasoning = 0.0
        time_merge = 0.0
        incremental_mode = "none"
        incremental_qids = 0

        if do_incremental and not changed_rule_ids:
            prev_rid = make_run_id(round_k - 1, "valid", cfg_id)
            prev_cands = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_candidates(prev_rid, partial=False)))
            _safe_copy(prev_cands, cands_full)
            if trace_enabled and trace_full is not None:
                prev_trace = maybe_windows_long_path(
                    os.path.join(ranked_rules_dir, _file_trace(prev_rid, graph, w_str, partial=False))
                )
                _safe_copy(prev_trace, trace_full)
            incremental_mode = "copy_prev"

        elif do_incremental and changed_rule_ids:
            prev_rid = make_run_id(round_k - 1, "valid", cfg_id)
            prev_cands = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_candidates(prev_rid, partial=False)))
            prev_trace = (
                maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_trace(prev_rid, graph, w_str, partial=False)))
                if trace_enabled
                else None
            )

            prev_conc = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_confidence_concrete(round_k - 1)))
            affected_rels = _collect_affected_relations(prev_conc, changed_rule_ids)
            if not affected_rels:
                do_incremental = False
            else:
                if trace_enabled and prev_trace is not None:
                    qids = _select_qids_from_trace(prev_trace, affected_rels)
                else:
                    split_np, _total_q = _ensure_valid_split()
                    qids = _select_qids_from_split_np(split_np, affected_rels)
                if not qids:
                    _safe_copy(prev_cands, cands_full)
                    if trace_enabled and prev_trace is not None and trace_full is not None:
                        _safe_copy(prev_trace, trace_full)
                    incremental_mode = "copy_prev"
                else:
                    qids_file = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_qids(rid)))
                    _safe_dump_json_compact(qids_file, qids)
                    incremental_mode = "qids"
                    incremental_qids = int(len(qids))

                    cands_part = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_candidates(rid, partial=True)))
                    trace_part = (
                        maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_trace(rid, graph, w_str, partial=True)))
                        if trace_enabled
                        else None
                    )

                    time_reasoning = _run(
                        _build_reasoning_cmd(
                            py,
                            dataset=args.dataset,
                            split="valid",
                            query_dir=str(args.query_dir),
                            rules_confidence_file=conf_file,
                            bgkg="all",
                            gpu=args.gpu,
                            top_k=args.top_k,
                            num_processes=args.num_processes,
                            window=args.window,
                            rule_lengths=args.rule_lengths,
                            evaluation_type=args.evaluation_type,
                            confidence_type=args.confidence_type,
                            min_conf=args.min_conf,
                            lmbda=args.lmbda,
                            weight_0=args.weight_0,
                            weight=args.weight,
                            coor_weight=args.coor_weight,
                            win_start=args.win_start,
                            is_sorted=args.is_sorted,
                            is_relax_time=args.is_relax_time,
                            is_sampled=args.is_sampled,
                            is_rule_priority=args.is_rule_priority,
                            trace_rules=str(args.trace_rules),
                            trace_top_rules=int(args.trace_top_rules),
                            graph_reasoning_type=graph,
                            rule_weight=w,
                            candidates_file_name=cands_part,
                            trace_file_name=trace_part if trace_enabled else None,
                            qids_file=qids_file,
                            bat_file_name=args.bat_file_name,
                            results_root_path=args.results_root_path,
                        )
                    )

                    t_merge0 = time.time()
                    if trace_enabled and prev_trace is not None:
                        total_queries = _count_non_empty_lines(prev_trace)
                    else:
                        _split_np, total_queries = _ensure_valid_split()
                    _merge_candidates(prev_cands, cands_part, cands_full, total_queries)
                    if trace_enabled and prev_trace is not None and trace_part is not None and trace_full is not None:
                        _merge_trace(prev_trace, trace_part, trace_full)
                    _safe_remove(cands_part)
                    if trace_part is not None:
                        _safe_remove(trace_part)
                    time_merge = float(time.time() - t_merge0)

        if not do_incremental:
            time_reasoning = _run(
                _build_reasoning_cmd(
                    py,
                    dataset=args.dataset,
                    split="valid",
                    query_dir=str(args.query_dir),
                    rules_confidence_file=conf_file,
                    bgkg="all",
                    gpu=args.gpu,
                    top_k=args.top_k,
                    num_processes=args.num_processes,
                    window=args.window,
                    rule_lengths=args.rule_lengths,
                    evaluation_type=args.evaluation_type,
                    confidence_type=args.confidence_type,
                    min_conf=args.min_conf,
                    lmbda=args.lmbda,
                    weight_0=args.weight_0,
                    weight=args.weight,
                    coor_weight=args.coor_weight,
                    win_start=args.win_start,
                    is_sorted=args.is_sorted,
                    is_relax_time=args.is_relax_time,
                    is_sampled=args.is_sampled,
                    is_rule_priority=args.is_rule_priority,
                    trace_rules=str(args.trace_rules),
                    trace_top_rules=int(args.trace_top_rules),
                    graph_reasoning_type=graph,
                    rule_weight=w,
                    candidates_file_name=cands_full,
                    trace_file_name=trace_full if trace_enabled else None,
                    qids_file=None,
                    bat_file_name=args.bat_file_name,
                    results_root_path=args.results_root_path,
                )
            )

        # evaluate(valid)
        time_evaluate = _run(
            _build_evaluate_cmd(
                py,
                dataset=args.dataset,
                split="valid",
                query_dir=str(args.query_dir),
                candidates_file_name=cands_full,
                graph_reasoning_type=graph,
                rule_weight=w,
                bat_file_name=args.bat_file_name,
                results_root_path=args.results_root_path,
            )
        )

        metrics_json = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_metrics(rid, graph, w_str, "json")))
        metrics = load_json(metrics_json, default={}) or {}
        picked = metrics.get("metrics", {}) if isinstance(metrics.get("metrics"), dict) else {}

        time_diag = 0.0
        evidence_cmd = [
            py,
            "build_llm_evidence.py",
            "-d",
            args.dataset,
            "--round",
            str(int(round_k)),
            "--test_data",
            "valid",
            "--graph_reasoning_type",
            graph,
            "--rule_weight",
            str(float(w)),
            "--candidates_file",
            os.path.basename(cands_full),
            "--confidence_concrete_file",
            conc_file,
            "--bat_file_name",
            args.bat_file_name,
            "--results_root_path",
            args.results_root_path,
        ]

        if trace_enabled and trace_full is not None:
            # trace-enabled diagnostics
            time_breakdown = _run(
                [
                    py,
                    "analyze_eval_breakdown.py",
                    "-d",
                    args.dataset,
                    "--test_data",
                    "valid",
                    "--candidates_file",
                    os.path.basename(cands_full),
                    "--graph_reasoning_type",
                    graph,
                    "--rule_weight",
                    str(float(w)),
                    "--trace_file",
                    os.path.basename(trace_full),
                    "--bat_file_name",
                    args.bat_file_name,
                    "--results_root_path",
                    args.results_root_path,
                ]
            )
            time_rel_diag = _run(
                [
                    py,
                    "diagnose_relations.py",
                    "-d",
                    args.dataset,
                    "--test_data",
                    "valid",
                    "--candidates_file",
                    os.path.basename(cands_full),
                    "--graph_reasoning_type",
                    graph,
                    "--rule_weight",
                    str(float(w)),
                    "--trace_file",
                    os.path.basename(trace_full),
                    "--min_n",
                    "50",
                    "--top_relations",
                    "50",
                    "--bat_file_name",
                    args.bat_file_name,
                    "--results_root_path",
                    args.results_root_path,
                ]
            )
            time_popbias = _run(
                [
                    py,
                    "diagnose_popularity_bias.py",
                    "-d",
                    args.dataset,
                    "--test_data",
                    "valid",
                    "--candidates_file",
                    os.path.basename(cands_full),
                    "--graph_reasoning_type",
                    graph,
                    "--rule_weight",
                    str(float(w)),
                    "--trace_file",
                    os.path.basename(trace_full),
                    "--min_n",
                    "50",
                    "--pop_top",
                    "15",
                    "--bat_file_name",
                    args.bat_file_name,
                    "--results_root_path",
                    args.results_root_path,
                ]
            )
            time_diag = float(time_breakdown + time_rel_diag + time_popbias)

            breakdown_json = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_breakdown(rid, graph, w_str, "json")))
            rel_diag_json = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_rel_diag(rid, graph, w_str, "json")))
            popbias_json = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_popbias(rid, graph, w_str, "json")))

            evidence_cmd += [
                "--trace_jsonl",
                os.path.basename(trace_full),
                "--breakdown_json",
                os.path.basename(breakdown_json),
                "--rel_diag_json",
                os.path.basename(rel_diag_json),
                "--popbias_json",
                os.path.basename(popbias_json),
            ]

        time_evidence = _run(evidence_cmd)

        evidence_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_evidence(rid, graph, w_str)))
        evidence_payload = load_json(evidence_path, default={}) or {}
        summary_payload = (
            evidence_payload.get("summary", {}) if isinstance(evidence_payload.get("summary", {}), dict) else {}
        )

        time_round = float(time.time() - t_round_start)
        return {
            "round": int(round_k),
            "split": "valid",
            "cfg_id": cfg_id,
            "run_id": rid,
            "confidence_file": conf_file,
            "confidence_concrete_file": conc_file,
            "candidates_file": os.path.basename(cands_full),
            "trace_file": os.path.basename(trace_full) if (trace_enabled and trace_full is not None) else "",
            "metrics_json": os.path.basename(metrics_json),
            "evidence_file": os.path.basename(evidence_path),
            "mrr": float(picked.get("mrr", 0.0)) if picked else 0.0,
            "hits1": float(picked.get("hits1", 0.0)) if picked else 0.0,
            "hits3": float(picked.get("hits3", 0.0)) if picked else 0.0,
            "hits10": float(picked.get("hits10", 0.0)) if picked else 0.0,
            "harm_rate": float(summary_payload.get("harm_rate", 0.0)),
            "NoCand": float(summary_payload.get("NoCand", 0.0)),
            "NoHit": float(summary_payload.get("NoHit", 0.0)),
            "HitTop1": float(summary_payload.get("HitTop1", 0.0)),
            "HitNotTop1": float(summary_payload.get("HitNotTop1", 0.0)),
            "incremental_mode": incremental_mode,
            "incremental_qids": int(incremental_qids),
            "time_reasoning_sec": round(float(time_reasoning), 3),
            "time_merge_sec": round(float(time_merge), 3),
            "time_evaluate_sec": round(float(time_evaluate), 3),
            "time_diag_sec": round(float(time_diag), 3),
            "time_evidence_sec": round(float(time_evidence), 3),
            "time_round_sec": round(float(time_round), 3),
        }

    # Round 0 eval + evidence
    summary_rows: List[Dict[str, Any]] = []
    row0 = run_valid_round(0, patch_path=None)
    row0["time_llm_sec"] = 0.0
    row0["time_apply_patch_sec"] = 0.0
    row0["time_iter_sec"] = float(row0.get("time_round_sec", 0.0))
    summary_rows.append(row0)

    iter_durations_sec: List[float] = []
    est_final_test_sec: Optional[float] = None
    if args.final_test == "Yes" or args.test_all_rounds == "Yes":
        try:
            per_test = float(row0.get("time_reasoning_sec", 0.0)) + float(row0.get("time_evaluate_sec", 0.0))
            n_test_runs = 0
            if args.test_all_rounds == "Yes":
                n_test_runs = int(args.K) + 1
            elif args.final_test == "Yes":
                n_test_runs = 1
            est_final_test_sec = float(per_test) * float(n_test_runs) if n_test_runs > 0 else None
        except Exception:
            est_final_test_sec = None

    # Rough ETA after warm-up round0 (uses round0 as proxy for one iteration).
    _print_eta(
        start_time=start_time,
        total_iters=int(args.K),
        done_iters=0,
        last_iters_sec=[float(row0.get("time_round_sec", 0.0))],
        est_final_test_sec=est_final_test_sec,
    )

    # Loop: k = 0..K-1
    for k in range(int(args.K)):
        prev = summary_rows[-1]
        evidence_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, str(prev.get("evidence_file"))))
        patch_cmd = [
            py,
            "llm_rule_governance.py",
            "--evidence_file",
            evidence_path,
            "--model_name",
            args.llm_model,
            "--allow_delete",
            str(args.allow_delete),
            "--allow_promote",
            str(args.allow_promote),
        ]
        if args.llm_dry_run:
            patch_cmd.append("--dry_run")
        time_llm = _run(patch_cmd)

        patch_path = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_patch(k + 1)))
        conf_in = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_confidence(k)))
        conc_in = maybe_windows_long_path(os.path.join(ranked_rules_dir, _file_confidence_concrete(k)))
        time_apply_patch = _run(
            [
                py,
                "apply_rule_patch.py",
                "--confidence_in",
                conf_in,
                "--confidence_concrete_in",
                conc_in,
                "--patch_file",
                patch_path,
                "--out_dir",
                ranked_rules_dir,
            ]
        )

        row = run_valid_round(k + 1, patch_path=patch_path)
        row["time_llm_sec"] = round(float(time_llm), 3)
        row["time_apply_patch_sec"] = round(float(time_apply_patch), 3)
        iter_sec = float(time_llm) + float(time_apply_patch) + float(row.get("time_round_sec", 0.0))
        row["time_iter_sec"] = round(float(iter_sec), 3)
        summary_rows.append(row)
        iter_durations_sec.append(float(iter_sec))

        _print_eta(
            start_time=start_time,
            total_iters=int(args.K),
            done_iters=int(len(iter_durations_sec)),
            last_iters_sec=iter_durations_sec,
            est_final_test_sec=est_final_test_sec,
        )

    # Select best round on VALID
    def key_fn(r: Dict[str, Any]) -> Tuple[float, float]:
        return (float(r.get("mrr", 0.0)), float(r.get("hits1", 0.0)))

    best = max(summary_rows, key=key_fn) if summary_rows else {"round": 0}
    best_round = int(best.get("round", 0))
    best_txt = (
        f"best_round={best_round}\n"
        f"criteria=valid_mrr_then_hits1\n"
        f"mrr={best.get('mrr', 0.0)} hits1={best.get('hits1', 0.0)}\n"
        f"cfg_id={cfg_id}\n"
        f"run_id={best.get('run_id')}\n"
        f"confidence_file={best.get('confidence_file')}\n"
        f"confidence_concrete_file={best.get('confidence_concrete_file')}\n"
        f"candidates_file={best.get('candidates_file')}\n"
        f"trace_file={best.get('trace_file')}\n"
        f"metrics_json={best.get('metrics_json')}\n"
        f"evidence_file={best.get('evidence_file')}\n"
    )
    safe_write_text(maybe_windows_long_path(os.path.join(ranked_rules_dir, "best_round.txt")), best_txt)

    # Save loop summary
    fieldnames = [
        "round",
        "split",
        "cfg_id",
        "run_id",
        "mrr",
        "hits1",
        "hits3",
        "hits10",
        "harm_rate",
        "NoCand",
        "NoHit",
        "HitTop1",
        "HitNotTop1",
        "incremental_mode",
        "incremental_qids",
        "confidence_file",
        "confidence_concrete_file",
        "candidates_file",
        "trace_file",
        "metrics_json",
        "evidence_file",
        "time_llm_sec",
        "time_apply_patch_sec",
        "time_reasoning_sec",
        "time_merge_sec",
        "time_evaluate_sec",
        "time_diag_sec",
        "time_evidence_sec",
        "time_round_sec",
        "time_iter_sec",
    ]
    safe_write_csv(maybe_windows_long_path(os.path.join(ranked_rules_dir, "loop_summary.csv")), summary_rows, fieldnames)

    # Test evaluation (optional)
    test_rows: List[Dict[str, Any]] = []
    if args.test_all_rounds == "Yes":
        for r in range(len(summary_rows)):
            rr = int(summary_rows[r].get("round", r))
            test_rows.append(run_test_round(rr))
        safe_write_csv(
            maybe_windows_long_path(os.path.join(ranked_rules_dir, "test_summary.csv")),
            test_rows,
            [
                "round",
                "split",
                "cfg_id",
                "run_id",
                "mrr",
                "hits1",
                "hits3",
                "hits10",
                "confidence_file",
                "candidates_file",
                "metrics_json",
                "metrics_csv",
            ],
        )

    # Final test (optional, best round) + stable "final_test_metrics.*"
    if args.final_test == "Yes":
        best_row = None
        if test_rows:
            for tr in test_rows:
                if int(tr.get("round", -1)) == int(best_round):
                    best_row = tr
                    break
        if best_row is None:
            best_row = run_test_round(best_round)

        best_metrics_json = maybe_windows_long_path(os.path.join(ranked_rules_dir, str(best_row.get("metrics_json"))))
        best_metrics_csv = maybe_windows_long_path(os.path.join(ranked_rules_dir, str(best_row.get("metrics_csv"))))
        m_payload = load_json(best_metrics_json, default={}) or {}
        safe_dump_json(maybe_windows_long_path(os.path.join(ranked_rules_dir, "final_test_metrics.json")), m_payload, indent=2)
        if best_metrics_csv and os.path.exists(best_metrics_csv):
            _safe_copy(best_metrics_csv, maybe_windows_long_path(os.path.join(ranked_rules_dir, "final_test_metrics.csv")))

    print(f"[Done] loop_summary => {maybe_windows_long_path(os.path.join(ranked_rules_dir, 'loop_summary.csv'))}")
    print(f"[Done] best_round   => {maybe_windows_long_path(os.path.join(ranked_rules_dir, 'best_round.txt'))}")


if __name__ == "__main__":
    main()
