import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

try:
    from utils_windows_long_path import maybe_windows_long_path, safe_open
except ImportError:  # pragma: no cover
    def maybe_windows_long_path(path):
        return path

    def safe_open(file_path, *args, **kwargs):
        return open(file_path, *args, **kwargs)


def _resolve_under_dataset(dataset_dir, path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    candidate = maybe_windows_long_path(os.path.join(dataset_dir, path))
    if os.path.exists(candidate):
        return candidate
    return path


def _read_relation_names(dataset_dir):
    relation2id_path = maybe_windows_long_path(os.path.join(dataset_dir, "relation2id.json"))
    if not os.path.exists(relation2id_path):
        return []
    with safe_open(relation2id_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.keys())
    return []


def _read_ts2id(dataset_dir, split_path):
    """
    Prefer dataset ts2id.json (consistent with Grapher). If missing, build a fallback mapping
    by sorting unique dates observed in split file.
    Returns: dict[str_date] -> int_ts_id
    """
    ts2id_path = maybe_windows_long_path(os.path.join(dataset_dir, "ts2id.json"))
    if os.path.exists(ts2id_path):
        try:
            with safe_open(ts2id_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): int(v) for k, v in data.items()}
        except Exception:
            pass

    dates = set()
    with safe_open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            date = parts[3].strip()
            if date:
                dates.add(date)

    dates_sorted = sorted(dates)  # YYYY-MM-DD lexicographic works
    return {d: i for i, d in enumerate(dates_sorted)}


def _read_matched_countries(matched_rows_path):
    """
    Build dict keyed by (Source_Name, Event_Text, Target_Name, Event_Date) -> (src_country, tgt_country).
    Missing/invalid rows are skipped. If duplicates exist, prefer non-empty countries.
    """
    matched = {}
    with safe_open(matched_rows_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            key = (
                (row.get("Source_Name") or "").strip(),
                (row.get("Event_Text") or "").strip(),
                (row.get("Target_Name") or "").strip(),
                (row.get("Event_Date") or "").strip(),
            )
            if not all(key):
                continue
            src_country = (row.get("Source_Country") or "").strip()
            tgt_country = (row.get("Target_Country") or "").strip()
            if key in matched:
                prev_src, prev_tgt = matched[key]
                if prev_src and prev_tgt:
                    continue
                if src_country and tgt_country:
                    matched[key] = (src_country, tgt_country)
            else:
                matched[key] = (src_country, tgt_country)
    return matched


def _reservoir_add(reservoir, seen_count, item, k, rng):
    """Reservoir sampling; seen_count should be AFTER increment."""
    if k <= 0:
        return
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = rng.randint(0, seen_count - 1)
    if j < k:
        reservoir[j] = item


def _percentile(sorted_vals, q):
    """
    q in [0,1]. Linear interpolation percentile (like numpy.quantile default).
    sorted_vals must be non-empty sorted list of floats.
    """
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo]) * (1 - frac) + float(sorted_vals[hi]) * frac


def build_relation_profiles(
    quadruples_path,
    matched_rows_path,
    *,
    dataset_dir,
    topk_pairs=10,
    repr_k=5,
    seed=0,
    all_relation_names=None,
):
    """
    Output per relation:
      - GeoBias: country_pairs_topk
      - TemporalActivity: level (LOW/MEDIUM/HIGH), count_train, span_steps, density, density_median
      - ReprEvents: reservoir sampled

    Notes:
      - count_train counts all facts in split (even if country unmatched).
      - GeoBias/ReprEvents only from matched_rows with valid countries.
      - span_steps uses ts2id.json if available else fallback mapping.
    """
    rng = random.Random(seed)
    matched = _read_matched_countries(matched_rows_path)
    ts2id = _read_ts2id(dataset_dir, quadruples_path)

    # Temporal raw stats
    rel_count = defaultdict(int)
    rel_min_ts = {}
    rel_max_ts = {}

    # GeoBias
    pair_counts = defaultdict(Counter)
    pair_total = defaultdict(int)

    # ReprEvents
    repr_events = defaultdict(list)
    repr_seen = defaultdict(int)

    relations_seen = set()

    with safe_open(quadruples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            src_name, rel, tgt_name, date = (p.strip() for p in parts)
            if not rel:
                continue

            relations_seen.add(rel)

            # temporal count + span
            rel_count[rel] += 1
            if date:
                ts_id = ts2id.get(date)
                if ts_id is not None:
                    if rel not in rel_min_ts or ts_id < rel_min_ts[rel]:
                        rel_min_ts[rel] = ts_id
                    if rel not in rel_max_ts or ts_id > rel_max_ts[rel]:
                        rel_max_ts[rel] = ts_id

            # geo + repr need matched countries
            countries = matched.get((src_name, rel, tgt_name, date))
            if not countries:
                continue
            src_country, tgt_country = countries
            if not src_country or not tgt_country:
                continue

            pair_total[rel] += 1
            pair_counts[rel][(src_country, tgt_country)] += 1

            repr_seen[rel] += 1
            event = {
                "date_ym": (date[:7] if date else None),
                "src_country": src_country,
                "tgt_country": tgt_country,
                "src_name": src_name,
                "tgt_name": tgt_name,
                "rel": rel,
            }
            _reservoir_add(repr_events[rel], repr_seen[rel], event, int(repr_k), rng)

    # target relations = all names from relation2id + seen in split
    if all_relation_names:
        relations_target = set(all_relation_names)
        relations_target.update(relations_seen)
    else:
        relations_target = set(relations_seen)

    # First pass: compute density list for quantiles/median
    rel_density = {}
    densities = []
    for rel in relations_target:
        count_train = int(rel_count.get(rel, 0))
        min_ts = rel_min_ts.get(rel)
        max_ts = rel_max_ts.get(rel)
        if min_ts is None or max_ts is None or max_ts < min_ts:
            rel_density[rel] = None
            continue
        span_steps = int(max_ts - min_ts)
        density = float(count_train) / float(span_steps + 1)
        rel_density[rel] = density
        densities.append(density)

    densities_sorted = sorted(densities)
    density_median = _percentile(densities_sorted, 0.5) if densities_sorted else None
    p33 = _percentile(densities_sorted, 0.3333333333) if densities_sorted else None
    p66 = _percentile(densities_sorted, 0.6666666667) if densities_sorted else None

    def to_level(d):
        if d is None or p33 is None or p66 is None:
            return None
        if d <= p33:
            return "LOW"
        if d <= p66:
            return "MEDIUM"
        return "HIGH"

    profiles = {}

    for rel in sorted(relations_target):
        # GeoBias
        total_pairs = int(pair_total.get(rel, 0))
        top_pairs = []
        if total_pairs > 0:
            items = sorted(pair_counts[rel].items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
            for (src_country, tgt_country), count in items[: int(topk_pairs)]:
                top_pairs.append(
                    {
                        "src": src_country,
                        "tgt": tgt_country,
                        "count": int(count),
                        "ratio": float(count) / float(total_pairs),
                    }
                )

        # TemporalActivity
        count_train = int(rel_count.get(rel, 0))
        min_ts = rel_min_ts.get(rel)
        max_ts = rel_max_ts.get(rel)
        span_steps = int(max_ts - min_ts) if (min_ts is not None and max_ts is not None and max_ts >= min_ts) else None
        density = rel_density.get(rel)
        level = to_level(density)

        profiles[rel] = {
            "geo_bias": {
                "country_pairs_topk": top_pairs,
                "country_pairs_total": total_pairs,
            },
            "temporal_activity": {
                "level": level,
                "count_train": count_train,
                "span_steps": span_steps,
                "density": density,
                "density_median": density_median,
                "p33": p33,
                "p66": p66,
            },
            "repr_events": repr_events.get(rel, []),
        }

    # Build inv_ profiles by flipping source/target fields.
    for rel in list(profiles.keys()):
        if rel.startswith("inv_"):
            continue
        inv_rel = f"inv_{rel}"
        if inv_rel in profiles:
            continue

        base = profiles.get(rel) or {}

        base_geo = base.get("geo_bias") or {}
        inv_pairs = []
        for entry in (base_geo.get("country_pairs_topk") or []):
            if not isinstance(entry, dict):
                continue
            inv_pairs.append(
                {
                    "src": entry.get("tgt"),
                    "tgt": entry.get("src"),
                    "count": entry.get("count"),
                    "ratio": entry.get("ratio"),
                }
            )

        inv_events = []
        for ev in (base.get("repr_events") or []):
            if not isinstance(ev, dict):
                continue
            inv_events.append(
                {
                    "date_ym": ev.get("date_ym"),
                    "src_country": ev.get("tgt_country"),
                    "tgt_country": ev.get("src_country"),
                    "src_name": ev.get("tgt_name"),
                    "tgt_name": ev.get("src_name"),
                    "rel": inv_rel,
                }
            )

        profiles[inv_rel] = {
            "geo_bias": {
                "country_pairs_topk": inv_pairs,
                "country_pairs_total": base_geo.get("country_pairs_total", 0),
            },
            "temporal_activity": base.get("temporal_activity") or {},
            "repr_events": inv_events,
        }

    return profiles


def main():
    parser = argparse.ArgumentParser(description="Offline builder for Relation Profile (semantic injection).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset, e.g., icews14")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"], help="Split name")
    parser.add_argument(
        "--matched",
        type=str,
        default="facts_matched_rows.txt",
        help="Matched rows TSV (facts_matched_rows.txt). If relative, resolved under dataset_dir.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help='Output JSON path. If omitted, uses results_root/bat/semantics/{dataset}/relation_profile.json',
    )
    parser.add_argument("--results_root_path", type=str, default="results", help="Used only for default --out")
    parser.add_argument("--bat_file_name", type=str, default="bat_file", help="Used only for default --out")
    parser.add_argument("--topk_pairs", type=int, default=10, help="Top-K country pairs to store per relation")
    parser.add_argument("--repr_k", type=int, default=5, help="Representative events to store per relation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for representative event sampling")
    args = parser.parse_args()

    dataset_dir = maybe_windows_long_path(os.path.join("datasets", args.dataset))
    split_path = maybe_windows_long_path(os.path.join(dataset_dir, f"{args.split}.txt"))
    matched_path = _resolve_under_dataset(dataset_dir, args.matched)

    args.results_root_path = args.results_root_path.strip('"')

    if args.out is None:
        args.out = maybe_windows_long_path(
            os.path.join(
                args.results_root_path,
                args.bat_file_name,
                "semantics",
                args.dataset,
                "relation_profile.json",
            )
        )

    relation_names = _read_relation_names(dataset_dir)
    profiles = build_relation_profiles(
        split_path,
        matched_path,
        dataset_dir=dataset_dir,
        topk_pairs=args.topk_pairs,
        repr_k=args.repr_k,
        seed=args.seed,
        all_relation_names=relation_names,
    )

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(maybe_windows_long_path(out_dir), exist_ok=True)

    with safe_open(out_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(profiles)} profiles to: {out_path}")


if __name__ == "__main__":
    main()
