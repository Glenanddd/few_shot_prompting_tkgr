import argparse
import os
from typing import Any, Dict, List, Tuple

from vlrg_utils import configure_stdout_utf8, load_json, round_tag, safe_dump_json, safe_write_text
from utils_windows_long_path import maybe_windows_long_path


def _iter_rules(conf_payload: Dict[str, Any]):
    for rel_key, rules in (conf_payload or {}).items():
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if isinstance(rule, dict):
                yield rel_key, rule


def _apply_actions_to_rule(rule: Dict[str, Any], delete_set: set, down: Dict[str, float], pro: Dict[str, float]):
    rid = str(rule.get("rule_id", "") or "")
    if not rid:
        return rule, False, False, False
    if rid in delete_set:
        return None, True, False, False  # deleted

    scaled = False
    promoted = False
    downed = False

    conf = float(rule.get("conf", rule.get("confidence", 0.0)))
    if rid in down:
        scale = float(down[rid])
        conf *= scale
        scaled = True
        downed = True
    if rid in pro:
        scale = float(pro[rid])
        conf *= scale
        scaled = True
        promoted = True

    # cap to [0, 1]
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0

    if scaled:
        # Keep both fields consistent; keep abstract_rule unchanged by spec.
        rule["conf"] = float(conf)
        rule["confidence"] = float(conf)

    return rule, False, downed, promoted


def _apply_patch(conf_payload: Dict[str, Any], patch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    actions = patch.get("actions") if isinstance(patch.get("actions"), dict) else {}
    delete_list = actions.get("delete", []) if isinstance(actions.get("delete", []), list) else []
    delete_set = set(str(x) for x in delete_list if x)

    down = actions.get("downweight") if isinstance(actions.get("downweight"), dict) else {}
    pro = actions.get("promote") if isinstance(actions.get("promote"), dict) else {}
    # Enforce spec ranges (clip):
    # - downweight: 0.2 ~ 0.9
    # - promote:    1.0 ~ 1.2
    down_s = {}
    for k, v in down.items():
        if k is None:
            continue
        try:
            vv = float(v)
        except Exception:
            continue
        vv = max(0.2, min(0.9, vv))
        down_s[str(k)] = float(vv)

    pro_s = {}
    for k, v in pro.items():
        if k is None:
            continue
        try:
            vv = float(v)
        except Exception:
            continue
        vv = max(1.0, min(1.2, vv))
        pro_s[str(k)] = float(vv)

    stats = {"deleted": 0, "downweighted": 0, "promoted": 0, "kept": 0}

    out: Dict[str, Any] = {}
    for rel_key, rules in (conf_payload or {}).items():
        if not isinstance(rules, list):
            continue
        new_rules = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            new_rule, deleted, downed, promoted = _apply_actions_to_rule(rule, delete_set, down_s, pro_s)
            if deleted:
                stats["deleted"] += 1
                continue
            if new_rule is None:
                stats["deleted"] += 1
                continue
            if downed:
                stats["downweighted"] += 1
            if promoted:
                stats["promoted"] += 1
            stats["kept"] += 1
            new_rules.append(new_rule)
        out[rel_key] = new_rules

    return out, stats


def main():
    configure_stdout_utf8()

    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence_in", required=True, type=str)
    parser.add_argument("--confidence_concrete_in", required=True, type=str)
    parser.add_argument("--patch_file", required=True, type=str)
    parser.add_argument("--out_dir", default="", type=str, help="Default: same dir as confidence_in")
    args = parser.parse_args()

    conf_in = maybe_windows_long_path(os.path.abspath(args.confidence_in))
    conc_in = maybe_windows_long_path(os.path.abspath(args.confidence_concrete_in))
    patch_path = maybe_windows_long_path(os.path.abspath(args.patch_file))
    out_dir = maybe_windows_long_path(os.path.abspath(args.out_dir)) if args.out_dir else maybe_windows_long_path(os.path.dirname(conf_in))

    conf_payload = load_json(conf_in, default=None)
    conc_payload = load_json(conc_in, default=None)
    patch = load_json(patch_path, default=None)
    if not isinstance(conf_payload, dict) or not isinstance(conc_payload, dict) or not isinstance(patch, dict):
        raise ValueError("Invalid inputs: confidence / confidence_concrete / patch must be JSON objects.")

    round_to = patch.get("round_to")
    if round_to is None:
        raise ValueError("patch_round*.json missing required field: round_to")
    round_to = int(round_to)

    os.makedirs(out_dir, exist_ok=True)
    rt = round_tag(round_to)
    out_conf = maybe_windows_long_path(os.path.join(out_dir, f"confidence_{rt}.json"))
    out_conc = maybe_windows_long_path(os.path.join(out_dir, f"confidence_concrete_{rt}.json"))
    out_log = maybe_windows_long_path(os.path.join(out_dir, f"patch_apply_log_{rt}.txt"))

    conf_out, stats_conf = _apply_patch(conf_payload, patch)
    conc_out, stats_conc = _apply_patch(conc_payload, patch)

    safe_dump_json(out_conf, conf_out, indent=2)
    safe_dump_json(out_conc, conc_out, indent=2)

    txt = (
        f"patch_file={os.path.basename(patch_path)}\n"
        f"confidence_in={os.path.basename(conf_in)}\n"
        f"confidence_concrete_in={os.path.basename(conc_in)}\n"
        f"confidence_out={os.path.basename(out_conf)}\n"
        f"confidence_concrete_out={os.path.basename(out_conc)}\n"
        "\n"
        f"deleted={stats_conf['deleted']} (conf) / {stats_conc['deleted']} (concrete)\n"
        f"downweighted={stats_conf['downweighted']} (conf) / {stats_conc['downweighted']} (concrete)\n"
        f"promoted={stats_conf['promoted']} (conf) / {stats_conc['promoted']} (concrete)\n"
        f"kept={stats_conf['kept']} (conf) / {stats_conc['kept']} (concrete)\n"
    )
    safe_write_text(out_log, txt)

    print(f"[Saved] {out_conf}")
    print(f"[Saved] {out_conc}")
    print(f"[Saved] {out_log}")


if __name__ == "__main__":
    main()
