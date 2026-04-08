import copy
import os
import argparse
import glob
import re
import shutil
import sys
import traceback
import multiprocessing as mp
import hashlib

import numpy as np

from grapher import Grapher
from params import str_to_bool
from rule_learning import Rule_Learner, normalize_var_constraints, rules_statistics
from temporal_walk import initialize_temporal_walk
from utils import save_json_data, load_json_data
from utils_windows_long_path import maybe_windows_long_path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================================================
# Parsing helpers
# =========================================================
def get_walk(rule, relation2id, inv_relation_id, regex):
    head_body = rule.split("<-")
    rule_head_full_name = head_body[0].strip()
    condition_string = head_body[1].strip()

    match = re.search(regex, rule_head_full_name)
    if match is None:
        raise ValueError(f"Rule head does not match relation_regex: {rule_head_full_name}")

    head_relation_name, head_subject, head_object, head_timestamp = match.groups()[:4]

    matches = re.findall(regex, condition_string)
    if not matches:
        raise ValueError(f"Rule body does not match relation_regex: {condition_string}")

    entities = (
        [head_object]
        + [m[1].strip() for m in matches[:-1]]
        + [matches[-1][1].strip(), matches[-1][2].strip()]
    )
    relation_ids = [relation2id[head_relation_name]] + [relation2id[m[0].strip()] for m in matches]

    # Reverse except the first element (keep original logic)
    entities = entities[:1] + entities[1:][::-1]
    relation_ids = relation_ids[:1] + [inv_relation_id[x] for x in relation_ids[:0:-1]]

    return {"entities": entities, "relations": relation_ids}


# =========================================================
# Reproducible seeding (per-rule deterministic seed)
# =========================================================
def derive_seed_32(base_seed: int, text: str) -> int:
    """
    Deterministically derive a 32-bit seed from (base_seed, text).
    Stable across runs & machines.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(base_seed)).encode("utf-8"))
    h.update(b"|")
    h.update(text.encode("utf-8"))
    v = int.from_bytes(h.digest(), "little", signed=False)
    return v % (2 ** 32)


# =========================================================
# Parallel worker globals + funcs (only in this file)
# =========================================================
_WORKER_RL = None
_WORKER_RELATION2ID = None
_WORKER_INV_RELATION_ID = None
_WORKER_RELATION_REGEX = None
_WORKER_RULES_VAR_DICT = None
_WORKER_IS_MERGE = False
_WORKER_IS_RELAX_TIME = False
_WORKER_BASE_SEED = 0  # 0 means non-reproducible mode


def _init_worker(edges, id2relation, inv_relation_id, dataset,
                 relation2id, relation_regex, rules_var_dict,
                 is_merge, is_relax_time, base_seed):
    """
    Each subprocess builds its own Rule_Learner (private, no shared state).

    Two modes:
    - base_seed == 0: non-reproducible (seed once per worker using urandom(4))
    - base_seed != 0: reproducible (we will reseed per-rule; worker init seed doesn't matter much,
                      but we still set something legal)
    """
    global _WORKER_RL, _WORKER_RELATION2ID, _WORKER_INV_RELATION_ID, _WORKER_RELATION_REGEX
    global _WORKER_RULES_VAR_DICT, _WORKER_IS_MERGE, _WORKER_IS_RELAX_TIME, _WORKER_BASE_SEED

    _WORKER_RELATION2ID = relation2id
    _WORKER_INV_RELATION_ID = inv_relation_id
    _WORKER_RELATION_REGEX = relation_regex
    _WORKER_RULES_VAR_DICT = rules_var_dict
    _WORKER_IS_MERGE = bool(is_merge)
    _WORKER_IS_RELAX_TIME = bool(is_relax_time)
    _WORKER_BASE_SEED = int(base_seed or 0)

    _WORKER_RL = Rule_Learner(edges, id2relation, inv_relation_id, dataset)

    if _WORKER_BASE_SEED == 0:
        # Non-reproducible: different seed per worker
        np.random.seed(int.from_bytes(os.urandom(4), "little", signed=False))
    else:
        # Reproducible mode: per-rule reseeding dominates; init with a fixed legal seed anyway
        np.random.seed(_WORKER_BASE_SEED % (2 ** 32))


def _rule_key(rule_dict):
    """
    Deterministic key used in main process to deduplicate rules across workers:
      (head_rel, body_rels tuple, var_constraints tuple)
    """
    head = int(rule_dict["head_rel"])
    body = tuple(int(x) for x in rule_dict["body_rels"])

    vc = rule_dict.get("var_constraints", [])
    vc_norm = []
    for x in vc:
        if isinstance(x, (list, tuple)):
            vc_norm.append(tuple(x))
        else:
            vc_norm.append((x,))
    vc_t = tuple(vc_norm)

    return (head, body, vc_t)


def _process_one_rule(rule_line, is_has_confidence):
    """
    Worker: parse one rule line and (optionally) compute confidence.

    Returns tuple:
      (parsed_ok, rule_key_or_none, rule_dict_or_none, rule_without_confidence_or_none, err_or_none)

    Reproducibility:
      - if _WORKER_BASE_SEED != 0:
          before processing each rule, reseed np.random with derived seed(base_seed, rule_text)
        This makes results independent of scheduling / chunksize / worker assignment.
    """
    global _WORKER_RL, _WORKER_RELATION2ID, _WORKER_INV_RELATION_ID, _WORKER_RELATION_REGEX
    global _WORKER_RULES_VAR_DICT, _WORKER_IS_MERGE, _WORKER_IS_RELAX_TIME, _WORKER_BASE_SEED

    try:
        line = (rule_line or "").strip()
        if not line:
            return (False, None, None, None, "empty line")

        if is_has_confidence:
            confidence = float(line.split("&")[-1].strip())
            temp_rule = line.split("&")[:-1]
            rule_without_confidence = "&".join(temp_rule).strip()
        else:
            confidence = 0.0
            rule_without_confidence = "&".join(line.split("&")).strip()

        # Per-rule deterministic seeding (strong reproducibility)
        if _WORKER_BASE_SEED != 0:
            np.random.seed(derive_seed_32(_WORKER_BASE_SEED, rule_without_confidence))

        # If merging and this rule exists in rules_var_dict, reuse it directly.
        if _WORKER_IS_MERGE and (_WORKER_RULES_VAR_DICT is not None):
            hit = _WORKER_RULES_VAR_DICT.get(rule_without_confidence)
            if hit is not None:
                rule_var = copy.deepcopy(hit)
                rule_var["llm_confidence"] = confidence
                k = _rule_key(rule_var)
                return (True, k, rule_var, rule_without_confidence, None)

        # Otherwise parse to walk and estimate confidence
        walk = get_walk(rule_without_confidence, _WORKER_RELATION2ID, _WORKER_INV_RELATION_ID, _WORKER_RELATION_REGEX)

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [_WORKER_INV_RELATION_ID[x] for x in walk["relations"][1:][::-1]]
        rule["var_constraints"] = _WORKER_RL.define_var_constraints(walk["entities"][1:][::-1])

        conf, rule_supp, body_supp = _WORKER_RL.estimate_confidence(rule, is_relax_time=_WORKER_IS_RELAX_TIME)
        rule["conf"] = conf
        rule["rule_supp"] = rule_supp
        rule["body_supp"] = body_supp
        rule["llm_confidence"] = confidence

        # parsed_ok True regardless; but only return rule_dict if it passes the gate
        if rule.get("conf", 0) or rule.get("llm_confidence", 0):
            k = _rule_key(rule)
            return (True, k, rule, rule_without_confidence, None)

        return (True, None, None, rule_without_confidence, "zero confidence")

    except Exception as e:
        return (False, None, None, None, repr(e))


def _process_one_rule_wrapper(args_tuple):
    rule_line, is_has_confidence = args_tuple
    return _process_one_rule(rule_line, is_has_confidence)

def dedup_rule_lines(lines, is_has_confidence=False):
    """
    Deduplicate rule lines by rule_without_confidence, keeping first occurrence order.

    - is_has_confidence=True: remove the last '&...'(confidence) part before dedup
    - is_has_confidence=False: dedup by stripped full line (still ignores trailing spaces)
    """
    seen = set()
    out = []
    for line in lines:
        s = (line or "").strip()
        if not s:
            continue

        if is_has_confidence:
            parts = s.split("&")
            if len(parts) >= 2:
                key = "&".join(parts[:-1]).strip()
            else:
                key = s
        else:
            key = s

        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out

# =========================================================
# Serial version (kept for fallback / debugging)
# =========================================================
def calculate_confidence(rule_path, relation2id, inv_relation_id, rl,
                         relation_regex, rules_var_dict, is_merge,
                         is_has_confidence=False, is_relax_time=False):
    llm_gen_rules_list = []
    rule_files = glob.glob(maybe_windows_long_path(os.path.join(rule_path, "rules.txt")))
    rules_by_file = []
    total_rules = 0
    for input_filepath in rule_files:
        with open(input_filepath, "r", encoding="utf-8") as f:
            rules = f.readlines()
        # 去重：按 rule_without_confidence
        rules = dedup_rule_lines(rules, is_has_confidence=is_has_confidence)
        rules_by_file.append((input_filepath, rules))
        total_rules += len(rules)

    progress = None
    use_fallback_progress = False
    processed = 0
    if total_rules > 0:
        if tqdm is not None:
            progress = tqdm(total=total_rules, desc="Overall rule progress", dynamic_ncols=True, file=sys.stdout)
        else:
            use_fallback_progress = True
            print(f"Overall rule progress: {processed}/{total_rules}", end="\r", flush=True)

    for _, rules in rules_by_file:
        for rule in rules:
            try:
                if is_has_confidence:
                    confidence = float(rule.split("&")[-1].strip())
                    temp_rule = rule.split("&")[:-1]
                    rule_without_confidence = "&".join(temp_rule).strip()
                else:
                    confidence = 0.0
                    rule_without_confidence = "&".join(rule.split("&")).strip()

                walk = get_walk(rule_without_confidence, relation2id, inv_relation_id, relation_regex)
                rl.create_rule_for_merge(
                    walk, confidence, rule_without_confidence, rules_var_dict,
                    is_merge, is_relax_time=is_relax_time
                )

                if not is_has_confidence:
                    llm_gen_rules_list.append(rule_without_confidence)

            except Exception as e:
                print(f"Error processing rule: {rule}")
                print(e)
                if getattr(rl, "debug", False):
                    traceback.print_exc()
            finally:
                if progress is not None:
                    progress.update(1)
                elif use_fallback_progress:
                    processed += 1
                    if processed == total_rules or processed % 100 == 0:
                        print(f"Overall rule progress: {processed}/{total_rules}", end="\r", flush=True)

    if progress is not None:
        progress.close()
    elif use_fallback_progress:
        print()

    return llm_gen_rules_list


# =========================================================
# Parallel version (main will use this by default)
# =========================================================
def calculate_confidence_parallel(rule_path, relation2id, inv_relation_id, rl_main,
                                  relation_regex, rules_var_dict, is_merge,
                                  dataset, id2relation, edges,
                                  is_has_confidence=False, is_relax_time=False,
                                  num_workers=0, chunksize=8,
                                  base_seed=0,
                                  stable_merge=False):
    """
    Parallelize over rules. Subprocesses compute rule dict; main process merges via rl_main.update_rules_dict().
    One global progress bar updated per completed rule.

    Reproducible mode:
      - base_seed != 0 => per-rule deterministic seeding (strong reproducibility).
    Stable merge:
      - stable_merge=True: collect all rule_dict first, sort by rule_key, then merge to ensure JSON list order stable.
    """
    llm_gen_rules_list = []

    rule_files = glob.glob(maybe_windows_long_path(os.path.join(rule_path, "rules.txt")))

    raw_lines = []
    for input_filepath in rule_files:
        with open(input_filepath, "r", encoding="utf-8") as f:
            raw_lines.extend(f.readlines())

    # ---- 去重：按 rule_without_confidence（忽略末尾 &confidence），保留首次出现顺序 ----
    tasks = dedup_rule_lines(raw_lines, is_has_confidence=is_has_confidence)
    total_rules = len(tasks)
    
    if total_rules == 0:
        return llm_gen_rules_list

    progress = None
    use_fallback_progress = False
    processed = 0
    if tqdm is not None:
        progress = tqdm(total=total_rules, desc="Overall rule progress", dynamic_ncols=True, file=sys.stdout)
    else:
        use_fallback_progress = True
        print(f"Overall rule progress: {processed}/{total_rules}", end="\r", flush=True)

    if not num_workers or num_workers <= 0:
        num_workers = max(1, (os.cpu_count() or 1))

    seen = set()
    collected = []  # for stable_merge

    ctx = mp.get_context("spawn")
    initargs = (
        edges,
        id2relation,
        inv_relation_id,
        dataset,
        relation2id,
        relation_regex,
        rules_var_dict,
        is_merge,
        is_relax_time,
        base_seed,
    )

    wrapper_tasks = [(r, is_has_confidence) for r in tasks]

    with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=initargs) as pool:
        it = pool.imap_unordered(_process_one_rule_wrapper, wrapper_tasks, chunksize=chunksize)

        for parsed_ok, k, rule_dict, rule_wo_conf, err in it:
            if progress is not None:
                progress.update(1)
            elif use_fallback_progress:
                processed += 1
                if processed == total_rules or processed % 100 == 0:
                    print(f"Overall rule progress: {processed}/{total_rules}", end="\r", flush=True)

            if (not is_has_confidence) and parsed_ok and rule_wo_conf:
                llm_gen_rules_list.append(rule_wo_conf)

            if rule_dict is None:
                continue

            if k is None:
                k = _rule_key(rule_dict)

            if k in seen:
                continue
            seen.add(k)

            if stable_merge:
                collected.append((k, rule_dict))
            else:
                rl_main.update_rules_dict(rule_dict)

    if stable_merge and collected:
        collected.sort(key=lambda x: x[0])
        for _, rd in collected:
            rl_main.update_rules_dict(rd)

    if progress is not None:
        progress.close()
    elif use_fallback_progress:
        print()

    return llm_gen_rules_list


# =========================================================
# Output helpers
# =========================================================
def normalize_canonical_rule_string(rule: str) -> str:
    """
    Normalize canonical rule string for stable hashing / IDs.

    Rules:
      - collapse repeated whitespace
      - normalize "<-" to " <- "
      - remove spaces around "&"
    """
    s = (rule or "").strip()
    s = re.sub(r"\s*<-\s*", " <- ", s)
    s = re.sub(r"\s*&\s*", "&", s)
    s = re.sub(r"\s+", " ", s)
    return s


def build_canonical_rule_string(rule: dict, id2relation: dict) -> str:
    """
    Build canonical rule string from a rule dict.
    Example:
      r(X0,X2,T2) <- b1(X0,X1,T0)&b2(X1,X2,T1)
    """
    body_len = len(rule.get("body_rels", []))
    var_constraints = [
        list(x) for x in normalize_var_constraints(rule.get("var_constraints", []), body_len)
    ]

    pos2var = {}
    for var_idx, positions in enumerate(var_constraints):
        for pos in positions:
            pos2var[int(pos)] = int(var_idx)

    head_obj_idx = pos2var.get(body_len, body_len)
    head = f"{id2relation[int(rule['head_rel'])]}(X0,X{head_obj_idx},T{body_len})"

    parts = []
    for i in range(body_len):
        sub_idx = pos2var.get(i, i)
        obj_idx = pos2var.get(i + 1, i + 1)
        parts.append(f"{id2relation[int(rule['body_rels'][i])]}(X{sub_idx},X{obj_idx},T{i})")

    canonical = head + " <- " + "&".join(parts)
    return normalize_canonical_rule_string(canonical)


def compute_rule_id(canonical_rule_string: str) -> str:
    canonical = normalize_canonical_rule_string(canonical_rule_string)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def build_abstract_rule_string(rule: dict, id2relation: dict) -> str:
    canonical = rule.get("rule") or build_canonical_rule_string(rule, id2relation)
    conf = float(rule.get("conf", 0.0))
    supp = int(rule.get("rule_supp", 0))
    body_supp = int(rule.get("body_supp", 0))
    return f"{conf:8.6f}  {supp:4d}  {body_supp:4d}  {canonical}"


def annotate_rules_with_ids(rules_dict: dict, id2relation: dict) -> None:
    """
    In-place annotation: add stable `rule_id`, `rule`(canonical string) and
    `confidence` field to each rule item.
    """
    for _, rules in (rules_dict or {}).items():
        for rule in rules:
            canonical = build_canonical_rule_string(rule, id2relation)
            rule["rule"] = canonical
            rule["rule_id"] = compute_rule_id(canonical)
            # Keep original fields; add alias fields required by governance/diagnostics.
            rule["confidence"] = float(rule.get("conf", 0.0))


def build_confidence_with_names(rules_dict, id2relation):
    output = {}
    for head_rel, rules in rules_dict.items():
        head_key = str(head_rel)
        output.setdefault(head_key, [])
        for rule in rules:
            item = copy.deepcopy(rule)
            item["head_rel_name"] = id2relation[rule["head_rel"]]
            item["body_rels_names"] = [id2relation[x] for x in rule["body_rels"]]
            # Required by VLRG: stable rule_id mapping + readable abstract rule string
            item["rule"] = item.get("rule") or build_canonical_rule_string(rule, id2relation)
            item["rule_id"] = item.get("rule_id") or compute_rule_id(item["rule"])
            item["confidence"] = float(item.get("conf", 0.0))
            item["supp"] = int(item.get("rule_supp", 0))
            item["body_supp"] = int(item.get("body_supp", 0))
            item["abstract_rule"] = build_abstract_rule_string(rule, id2relation)
            output[head_key].append(item)
    return output

# =========================================================
# Main
# =========================================================
def main(args):
    is_merge = args.is_merge
    dataset_dir = maybe_windows_long_path(os.path.join(".", "datasets", args.dataset))
    data = Grapher(dataset_dir)

    temporal_walk = initialize_temporal_walk(args.bgkg, data, args.transition_distr)
    rl = Rule_Learner(temporal_walk.edges, data.id2relation, data.inv_relation_id, args.dataset)

    args.results_root_path = args.results_root_path.strip('"')
    if args.random_walk_rules == True:
        final_summary_rule_path = maybe_windows_long_path(
        os.path.join(args.results_root_path, args.bat_file_name, "sampled_path", args.dataset)
    )
    else:
        final_summary_rule_path = maybe_windows_long_path(
            os.path.join(args.results_root_path, args.bat_file_name, "gen_rules_iteration", args.dataset, "final_summary")
        )

    constant_config = load_json_data(maybe_windows_long_path(os.path.join("Config", "constant.json")))
    relation_regex = constant_config["relation_regex"][args.dataset]

    rules_var_path = maybe_windows_long_path(
        os.path.join(args.results_root_path, args.bat_file_name, "sampled_path", args.dataset, "original", "rules_var.json")
    )
    rules_var_dict = load_json_data(rules_var_path)

    use_parallel = (args.num_workers is None) or (args.num_workers == 0) or (args.num_workers > 1)

    if use_parallel:
        llm_gen_rules_list = calculate_confidence_parallel(
            final_summary_rule_path,
            data.relation2id,
            data.inv_relation_id,
            rl,
            relation_regex,
            rules_var_dict,
            is_merge,
            dataset=args.dataset,
            id2relation=data.id2relation,
            edges=temporal_walk.edges,
            is_relax_time=args.is_relax_time,
            is_has_confidence=False,
            num_workers=args.num_workers,
            chunksize=args.chunksize,
            base_seed=args.base_seed,
            stable_merge=args.stable_merge,
        )
    else:
        # Serial run: if you also want reproducible serial, you can do np.random.seed(args.base_seed) here.
        if args.base_seed:
            np.random.seed(args.base_seed % (2 ** 32))
        llm_gen_rules_list = calculate_confidence(
            final_summary_rule_path,
            data.relation2id,
            data.inv_relation_id,
            rl,
            relation_regex,
            rules_var_dict,
            is_merge,
            is_relax_time=args.is_relax_time,
        )

    save_rules(args, rules_var_dict, rl, llm_gen_rules_list, is_merge, data)



def save_rules(args, rules_var_dict, rl, llm_gen_rules_list, is_merge, data):
    dir_path = maybe_windows_long_path(
        os.path.join(args.results_root_path, args.bat_file_name, args.output_path, args.dataset)
    )

    os.makedirs(dir_path, exist_ok=True)

    if args.is_only_with_original_rules:
        confidence_file_name = "original_confidence.json"
    else:
        if is_merge:
            original_rules_set = set(rules_var_dict)
            llm_gen_rules_set = set(llm_gen_rules_list)
            for rule_chain in original_rules_set - llm_gen_rules_set:
                rule = rules_var_dict[rule_chain]
                rl.update_rules_dict(rule)

            confidence_file_name = "merge_confidence.json"
        else:
            confidence_file_name = "confidence.json"

    # Added for subset prompt-eval (2026-01-20)
    # Filter AFTER merge/stable_merge, BEFORE saving confidence.json to avoid re-injecting non-selected relations.
    selected_relations_path = maybe_windows_long_path(
        os.path.join(args.results_root_path, args.bat_file_name,"sampled_path", args.dataset, "selected_relations.json")
    )
    if args.selected_relations is True and os.path.exists(selected_relations_path):
        selected_payload = load_json_data(selected_relations_path) or {}
        keep_ids = set(int(x) for x in (selected_payload.get("selected_head_rel_ids", []) or []))
        rl.rules_dict = {k: v for k, v in rl.rules_dict.items() if int(k) in keep_ids}
        print(f"[Subset] Filter rl.rules_dict by keep_ids: {len(rl.rules_dict)} relations kept")

    # VLRG: attach stable rule_id + canonical rule string to each rule.
    annotate_rules_with_ids(rl.rules_dict, data.id2relation)

    rules_statistics(rl.rules_dict)

    confidence_file_path = maybe_windows_long_path(os.path.join(dir_path, confidence_file_name))
    save_json_data(rl.rules_dict, confidence_file_path)

    confidence_concrete_path = maybe_windows_long_path(os.path.join(dir_path, "confidence_concrete.json"))
    confidence_with_names = build_confidence_with_names(rl.rules_dict, data.id2relation)
    save_json_data(confidence_with_names, confidence_concrete_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="family")
    parser.add_argument(
        "--model_name",
        default="none",
        help="model name",
        choices=["none", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
    )
    parser.add_argument("--input_path", default="clean_rules", type=str, help="input folder")
    parser.add_argument("--output_path", default="ranked_rules", type=str, help="path to output file")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--transition_distr", default="exp", type=str)
    parser.add_argument("--is_merge", default="no", type=str_to_bool)
    parser.add_argument("--is_only_with_original_rules", default="no", type=str_to_bool)
    parser.add_argument("--is_iteration", default="yes", type=str_to_bool)
    parser.add_argument("--bgkg", default="test", type=str, choices=["train", "valid", "train_valid", "all", "test"])
    parser.add_argument("--is_relax_time", default="no", type=str_to_bool)
    parser.add_argument("--bat_file_name", type=str, default="bat_file", help="Batch file name")
    parser.add_argument(
        "--results_root_path",
        type=str,
        default="results",
        help='Results root path. Must put this parameter at last position on cmd line to avoid parsing error.',
    )

    # Parallel controls (only affect this file)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="0=auto(cpu_count), 1=serial, >1 parallel processes",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="multiprocessing imap chunksize (bigger=less overhead)",
    )

    # Reproducibility controls
    parser.add_argument(
        "--base_seed",
        type=int,
        default=1,
        help="0 disables reproducibility; non-zero enables per-rule deterministic seeding",
    )
    parser.add_argument(
        "--stable_merge",
        default="no",
        type=str_to_bool,
        help="yes => sort results before merging to keep output order stable",
    )
    parser.add_argument(
        "--random_walk_rules",
        default="no",
        type=str_to_bool,
        help="use random walk sampled rules",
    )
    parser.add_argument("--selected_relations", type=str_to_bool,
                        default='no', help="Enable selected relations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
