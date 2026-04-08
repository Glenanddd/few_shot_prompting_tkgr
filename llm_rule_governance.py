import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from utils_windows_long_path import maybe_windows_long_path
from llms import get_registed_model
from vlrg_utils import configure_stdout_utf8, load_json, round_tag, safe_dump_json, safe_write_text


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Try strict JSON parse first; then fall back to extracting the first {...} block.
    """
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Best-effort: extract a JSON object substring
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _validate_and_sanitize_patch(
    patch: Dict[str, Any],
    *,
    round_from: int,
    max_delete: int,
    max_downweight: int,
    max_promote: int,
    downweight_min: float,
    downweight_max: float,
    promote_max: float,
    based_on_split: str,
    allowed_rule_ids: Optional[set] = None,
) -> Dict[str, Any]:
    actions = patch.get("actions") if isinstance(patch.get("actions"), dict) else {}

    delete_list = actions.get("delete", [])
    if not isinstance(delete_list, list):
        delete_list = []
    delete_list = [str(x) for x in delete_list if x]
    if allowed_rule_ids is not None:
        delete_list = [x for x in delete_list if x in allowed_rule_ids]
    delete_list = delete_list[: int(max_delete)]

    down = actions.get("downweight", {})
    if not isinstance(down, dict):
        down = {}
    down_s: Dict[str, float] = {}
    for k, v in down.items():
        if k is None:
            continue
        try:
            vv = float(v)
        except Exception:
            continue
        vv = max(float(downweight_min), min(float(downweight_max), vv))
        rid = str(k)
        if allowed_rule_ids is not None and rid not in allowed_rule_ids:
            continue
        down_s[rid] = float(vv)
        if len(down_s) >= int(max_downweight):
            break

    pro_s: Dict[str, float] = {}
    if int(max_promote) > 0:
        pro = actions.get("promote", {})
        if not isinstance(pro, dict):
            pro = {}
        for k, v in pro.items():
            if k is None:
                continue
            try:
                vv = float(v)
            except Exception:
                continue
            vv = max(1.0, min(float(promote_max), vv))
            rid = str(k)
            if allowed_rule_ids is not None and rid not in allowed_rule_ids:
                continue
            pro_s[rid] = float(vv)
            if len(pro_s) >= int(max_promote):
                break

    # Delete wins over others
    delete_set = set(delete_list)
    for rid in list(down_s.keys()):
        if rid in delete_set:
            down_s.pop(rid, None)
    for rid in list(pro_s.keys()):
        if rid in delete_set:
            pro_s.pop(rid, None)

    return {
        "round_from": int(round_from),
        "round_to": int(round_from) + 1,
        "based_on_split": str(based_on_split),
        "actions": {
            "delete": delete_list,
            "downweight": down_s,
            "promote": pro_s,
        },
        "notes": str(patch.get("notes", "") or "").strip()[:2000],
    }


def main():
    configure_stdout_utf8()

    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_file", required=True, type=str, help="Path to round{k}_evidence_for_llm.json")
    # Keep compatible with Iteration_reasoning.py / llms/chatgpt.py (OpenRouter)
    parser.add_argument("--model_name", default="gpt-5-nano", type=str)
    parser.add_argument("--allow_delete", default="Yes", type=str, choices=["Yes", "No"])
    parser.add_argument("--allow_promote", default="Yes", type=str, choices=["Yes", "No"])
    parser.add_argument("--max_actions_delete", default=100, type=int)
    parser.add_argument("--max_actions_downweight", default=200, type=int)
    parser.add_argument("--max_actions_promote", default=50, type=int)
    parser.add_argument("--dry_run", action="store_true", help="Write an empty patch without calling the API")

    # Add model-specific args (e.g., --retry/--or_rps/--or_timeout) like Iteration_reasoning.py.
    args_pre, _ = parser.parse_known_args()
    LLM = get_registed_model(args_pre.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    evidence_path = os.path.abspath(args.evidence_file)
    evidence = load_json(evidence_path, default=None)
    if not isinstance(evidence, dict):
        raise ValueError(f"Invalid evidence JSON: {evidence_path}")

    meta = evidence.get("meta") if isinstance(evidence.get("meta"), dict) else {}
    round_from = int(meta.get("round", 0))
    based_on_split = str(meta.get("split", "valid"))
    trace_enabled = bool(meta.get("trace_enabled", True))

    allow_delete = str(getattr(args, "allow_delete", "Yes")).strip().lower() == "yes"
    allow_promote = str(getattr(args, "allow_promote", "Yes")).strip().lower() == "yes"

    allowed_rule_ids = set()
    for r in (evidence.get("harmful_rules") or []):
        if isinstance(r, dict) and r.get("rule_id"):
            allowed_rule_ids.add(str(r.get("rule_id")))
    for c in (evidence.get("cases") or []):
        if not isinstance(c, dict):
            continue
        for e in (c.get("top1_rules") or []):
            if isinstance(e, dict) and e.get("rule_id") and str(e.get("rule_id")) != "GRAPH":
                allowed_rule_ids.add(str(e.get("rule_id")))
        for e in (c.get("ans_rules") or []):
            if isinstance(e, dict) and e.get("rule_id") and str(e.get("rule_id")) != "GRAPH":
                allowed_rule_ids.add(str(e.get("rule_id")))

    allowed = evidence.get("allowed_actions") if isinstance(evidence.get("allowed_actions"), dict) else {}
    down_cfg = allowed.get("downweight") if isinstance(allowed.get("downweight"), dict) else {}
    pro_cfg = allowed.get("promote") if isinstance(allowed.get("promote"), dict) else {}
    down_min = float(down_cfg.get("min_scale", 0.2))
    down_max = float(down_cfg.get("max_scale", 0.9))
    pro_max = float(pro_cfg.get("max_scale", 1.2))

    out_dir = os.path.dirname(evidence_path)
    round_to = int(round_from) + 1
    rt = round_tag(round_to)
    out_path = maybe_windows_long_path(os.path.join(out_dir, f"patch_{rt}.json"))
    prompt_path = maybe_windows_long_path(os.path.join(out_dir, f"llm_prompt_{rt}.txt"))
    output_path = maybe_windows_long_path(os.path.join(out_dir, f"llm_output_{rt}.txt"))
    usage_path = maybe_windows_long_path(os.path.join(out_dir, f"llm_usage_{rt}.json"))

    max_del = int(args.max_actions_delete) if allow_delete else 0
    max_pro = int(args.max_actions_promote) if allow_promote else 0

    if args.dry_run:
        patch = {
            "round_from": round_from,
            "round_to": round_from + 1,
            "based_on_split": based_on_split,
            "actions": {"delete": [], "downweight": {}, "promote": {}},
            "notes": "dry_run",
        }
        safe_dump_json(out_path, patch, indent=2)
        safe_write_text(prompt_path, "dry_run\n")
        safe_write_text(output_path, "dry_run\n")
        print(f"[Saved] {out_path}")
        return

    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not or_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Please set it in your environment (or use --dry_run)."
        )

    allowed_actions_desc = "downweight"
    if allow_delete:
        allowed_actions_desc = "delete / downweight" if not allow_promote else "delete / downweight / promote"
    else:
        allowed_actions_desc = "downweight" if not allow_promote else "downweight / promote"

    evidence_level_note = ""
    if not trace_enabled:
        evidence_level_note = (
            "EVIDENCE LEVEL: aggregate-only (no trace attribution).\n"
            "- cases/top1_rules/ans_rules are missing by design.\n"
            "- harmful_rules are heuristic suspects (worst_relations-based), NOT attribution-based.\n"
            "Be conservative: prefer fewer/smaller downweights; avoid aggressive deletions.\n"
        )

    schema = (
        "{\n"
        f"  \"round_from\": {round_from},\n"
        f"  \"round_to\": {round_from + 1},\n"
        f"  \"based_on_split\": \"{based_on_split}\",\n"
        "  \"actions\": {\n"
        "    \"delete\": [\"rule_id\", ...],\n"
        "    \"downweight\": {\"rule_id\": 0.6, ...},\n"
        "    \"promote\": {\"rule_id\": 1.05, ...}\n"
        "  },\n"
        "  \"notes\": \"optional\"\n"
        "}\n"
    )

    system_lines = [
        "You are a rule governance agent for temporal KG rule reasoning.",
        "You must ONLY output a single valid JSON object, with no extra text.",
        f"Allowed actions: {allowed_actions_desc}. Do NOT invent new rules.",
        "Do NOT use entity names, countries, or world knowledge; ONLY use the provided evidence and abstract_rule strings.",
        "Avoid overfitting single cases; prioritize consistent statistical signals.",
    ]
    if evidence_level_note:
        system_lines.append(evidence_level_note.rstrip("\n"))
    system_lines.append("Decision criteria:")
    if trace_enabled:
        system_lines += [
            "- If a rule has high harm_rate and low help_top1 => prefer delete.",
            "- If a rule correlates with popularity bias => prefer downweight.",
            "- If a rule often supports GT but is drowned out => small promote.",
        ]
    else:
        system_lines.append(
            "- Use summary + worst_relations to identify problematic relations, then downweight heuristic suspect rules."
        )
    system_lines.append(f"Downweight scale range: [{down_min}, {down_max}]. Promote scale range: [1.0, {pro_max}].")
    system_lines.append("The JSON schema you must output:")
    system_prompt = "\n".join(system_lines) + "\n" + schema

    # NOTE: llms/chatgpt.py currently sends a single user message; embed system+user instruction together.
    combined_prompt = (
        system_prompt
        + "\n"
        + "Generate a compact governance patch JSON based on this evidence.\n"
        f"Constraints:\n"
        f"- delete <= {int(max_del)} rules\n"
        f"- downweight <= {int(args.max_actions_downweight)} rules\n"
        f"- promote <= {int(max_pro)} rules\n"
        "- Prefer fewer actions over many weak actions.\n"
        "- Do not reference any rule_id that is not present in evidence.harmful_rules.\n"
        "\n"
        "EVIDENCE_JSON:\n"
        + json.dumps(evidence, ensure_ascii=False)
    )

    # Persist prompt for reproducibility/debugging.
    safe_write_text(prompt_path, combined_prompt)

    model = LLM(args)
    model.prepare_for_inference()

    generated = model.generate_sentence(combined_prompt, return_usage=True)
    usage_payload = None
    if isinstance(generated, tuple) and len(generated) == 2:
        content = (generated[0] or "").strip()
        usage_payload = generated[1]
    else:
        content = str(generated or "").strip()

    # Persist raw model output (even if it is not valid JSON).
    safe_write_text(output_path, content + "\n")
    if isinstance(usage_payload, dict):
        safe_dump_json(usage_path, usage_payload, indent=2)
    parsed = _extract_json_object(content)
    if parsed is None:
        raise ValueError(f"Model output is not valid JSON. Raw output:\n{content[:2000]}")

    patch = _validate_and_sanitize_patch(
        parsed,
        round_from=round_from,
        max_delete=int(max_del),
        max_downweight=int(args.max_actions_downweight),
        max_promote=int(max_pro),
        downweight_min=down_min,
        downweight_max=down_max,
        promote_max=pro_max,
        based_on_split=based_on_split,
        allowed_rule_ids=allowed_rule_ids,
    )

    safe_dump_json(out_path, patch, indent=2)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
