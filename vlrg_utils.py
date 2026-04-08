import csv
import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils_windows_long_path import maybe_windows_long_path, safe_open


def configure_stdout_utf8() -> None:
    """
    Avoid Windows console UnicodeEncodeError (e.g., GBK).
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_ranked_rules_dir(results_root_path: str, bat_file_name: str, dataset: str) -> str:
    results_root_path = (results_root_path or "results").strip('"')
    return maybe_windows_long_path(os.path.join(results_root_path, bat_file_name, "ranked_rules", dataset))


def safe_dump_json(path: str, obj: Any, *, indent: int = 2) -> None:
    try:
        with safe_open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
    except PermissionError as e:
        raise PermissionError(
            f"PermissionError writing JSON: {path}\n"
            "This can happen if the file is opened in Excel or another program. "
            "Please close it or change output name."
        ) from e


def safe_write_text(path: str, text: str) -> None:
    try:
        with safe_open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(text)
    except PermissionError as e:
        raise PermissionError(
            f"PermissionError writing text: {path}\n"
            "This can happen if the file is opened in Excel or another program. "
            "Please close it or change output name."
        ) from e


def safe_write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    try:
        with safe_open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
    except PermissionError as e:
        raise PermissionError(
            f"PermissionError writing CSV: {path}\n"
            "This can happen if the file is opened in Excel or another program. "
            "Please close it or change output name."
        ) from e


def load_json(path: str, default: Any = None) -> Any:
    try:
        if not os.path.exists(maybe_windows_long_path(path)):
            return default
        with safe_open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with safe_open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            yield json.loads(s)


def percentile_from_sorted(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = float(sorted_vals[f]) * (c - k)
    d1 = float(sorted_vals[c]) * (k - f)
    return d0 + d1


def topk_counter(counter: Dict[int, int], k: int) -> List[Dict[str, int]]:
    items = sorted(counter.items(), key=lambda x: (-int(x[1]), int(x[0])))
    return [{"eid": int(eid), "cnt": int(cnt)} for eid, cnt in items[: int(k)]]


def stable_json_dumps(obj: Any) -> str:
    """
    Deterministic JSON string for hashing / run IDs.
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cfg_id_from_config(cfg: Dict[str, Any], *, length: int = 8) -> str:
    """
    cfg_id = sha1(stable_json_dumps(cfg))[:length]
    """
    s = stable_json_dumps(cfg)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[: int(length)]


def round_tag(round_k: int, *, width: int = 3) -> str:
    return f"round_{int(round_k):0{int(width)}d}"


def run_id(round_k: int, split: str, cfg_id: str) -> str:
    return f"{round_tag(round_k)}_{str(split)}_cfg{str(cfg_id)}"


def candidates_run_id(candidates_file: str) -> str:
    """
    Preferred candidate filename: cands_{run_id}.json
    Returns {run_id} if matches; else returns the basename without extension.
    """
    base = os.path.splitext(os.path.basename(str(candidates_file)))[0]
    if base.startswith("cands_"):
        return base[len("cands_") :]
    return base


def fmt_float_for_name(x: float, *, decimals: int = 4) -> str:
    """
    Stable float formatting for filenames: trim trailing zeros.
    """
    s = f"{float(x):.{int(decimals)}f}".rstrip("0").rstrip(".")
    return "0" if s in {"", "-0"} else s
