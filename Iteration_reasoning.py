import argparse
import glob
import os.path
import sys
import traceback
import time
import copy
import itertools
import shutil
import threading
import random
from difflib import get_close_matches
from datetime import datetime

from tqdm import tqdm
from functools import partial

from data import *
from multiprocessing.pool import ThreadPool
from utils import *
from utils_windows_long_path import *
from llms import get_registed_model
from grapher import Grapher
from temporal_walk import Temporal_Walk
from rule_learning import Rule_Learner, rules_statistics, verbalize_rule
import tiktoken

from concurrent.futures import ThreadPoolExecutor
from params import str_to_bool
from utils_method_1_example_selection import format_examples, format_examples_ex1, load_example_pool_index, retrieve_dynamic_examples
from utils_method_2_semantics_enhance import build_profile_block, load_relation_profiles
import hashlib
import multiprocessing as mp
import io
from concurrent.futures import as_completed


def read_paths(path):
    results = []
    with safe_open(path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def build_prompt_for_common(head, prompt_dict):
    definition = prompt_dict['common']['definition'].format(head=head)
    role = prompt_dict['common']['role'].format(head=head)
    examples_title = prompt_dict['common']['examples_title']
    examples = prompt_dict['common']['examples']
    sample_paths_title = prompt_dict['common']['sample_paths_title']
    goal = prompt_dict['common']['goal'].format(head=head)
    candidate = prompt_dict['common']['candidate']
    return_rules = prompt_dict['common']['return_rules']
    return definition , role , examples_title , examples, sample_paths_title , goal , candidate, return_rules

def build_prompt_for_common_ex1(head, prompt_dict):
    definition = prompt_dict['common_ex1']['definition'].format(head=head)
    role = prompt_dict['common_ex1']['role'].format(head=head)
    examples_title = prompt_dict['common_ex1']['examples_title']
    examples = prompt_dict['common_ex1']['examples']
    sample_random_walk_paths_title = prompt_dict['common_ex1']['sample_random_walk_paths_title']
    candidate = prompt_dict['common_ex1']['candidate']
    rule_head_title_for_user = prompt_dict['common_ex1']['rule_head_title_for_user'].format(head=head)
    generated_temporal_logic_rules_title = prompt_dict['common_ex1']['generated_temporal_logic_rules_title']
    return definition , role , examples_title , examples, sample_random_walk_paths_title , candidate, rule_head_title_for_user, generated_temporal_logic_rules_title

def build_prompt_for_zero(head, prompt_dict):
    definition = prompt_dict['zero']['definition'].format(head=head)
    role = prompt_dict['zero']['role'].format(head=head)
    examples_title = prompt_dict['zero']['examples_title']
    examples = prompt_dict['zero']['examples']
    goal = prompt_dict['zero']['goal'].format(head=head)
    candidate = prompt_dict['zero']['candidate']
    return_rules = prompt_dict['zero']['return_rules']
    return  definition, role , examples_title , examples , goal , candidate, return_rules


def build_prompt_based_high(head, candidate_rels, is_zero, args, prompt_dict):
    # head = clean_symbol_in_rel(head)
    chain_defination = prompt_dict['chain_defination_for_high'].format(
        head=head)

    context = prompt_dict['iteration_context_for_high'].format(head=head)

    high_quality_context = prompt_dict['example_for_high']
    # predict = prompt_dict['interaction_finale_predict_for_high'].format(head=head, k=20)
    predict = prompt_dict['interaction_finale_predict_for_high'].format(
        head=head)
    return_rules = prompt_dict['return_for_high']

    return chain_defination + context + high_quality_context, predict, return_rules


def build_prompt_based_low(head, candidate_rels, is_zero, args, prompt_dict):
    # head = clean_symbol_in_rel(head)
    definition = prompt_dict['iteration']['definition'].format(head=head)
    role = prompt_dict['iteration']['role']
    examples_title = prompt_dict['iteration']['examples_title']
    examples = prompt_dict['iteration']['examples']
    low_quality_rules_title = prompt_dict['iteration']['low_quality_rules_title']
    sample_rules_title = prompt_dict['iteration']['sample_rules_title']
    goal = prompt_dict['iteration']['goal'].format(head=head)
    candidate = prompt_dict['iteration']['candidate']
    return_rules = prompt_dict['iteration']['return_rules']
    return definition , role , examples_title , examples, low_quality_rules_title, sample_rules_title , goal , candidate, return_rules


def init_usage_summary():
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "elapsed": 0.0,
    }


def merge_usage_summary(accumulator, usage):
    if not usage:
        return accumulator
    accumulator["calls"] += usage.get("calls", 0)
    accumulator["prompt_tokens"] += usage.get("prompt_tokens", 0)
    accumulator["completion_tokens"] += usage.get("completion_tokens", 0)
    accumulator["total_tokens"] += usage.get("total_tokens", 0)
    accumulator["cost"] += usage.get("cost", 0.0) or 0.0
    accumulator["elapsed"] += usage.get("elapsed", 0.0) or 0.0
    return accumulator


def collect_usage_from_generation(model, prompt, response_text, meta=None):
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    cost = 0.0
    elapsed = 0.0
    if meta:
        prompt_tokens = meta.get("prompt_tokens")
        completion_tokens = meta.get("completion_tokens")
        total_tokens = meta.get("total_tokens")
        cost = meta.get("cost", 0.0) or 0.0
        elapsed = meta.get("elapsed", 0.0) or 0.0

    if prompt_tokens is None:
        prompt_tokens = model.token_len(prompt)
    if completion_tokens is None and response_text is not None:
        completion_tokens = model.token_len(response_text)
    if completion_tokens is None:
        completion_tokens = 0
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "elapsed": elapsed,
        "calls": 1,
    }


def format_tokens(num):
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    if num >= 1_000:
        return f"{num/1_000:.1f}K"
    return f"{num:.0f}"


def format_seconds(seconds):
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{sec}s")
    return "".join(parts)


def log_stage(message, idx=None, total=None):
    prefix = f"[{datetime.now().strftime('%H:%M:%S')}] "
    if idx is not None and total is not None:
        print(f"\n{prefix}[{idx}/{total}] {message}")
    else:
        print(f"\n{prefix}{message}")


def update_progress_bar(progress, stats, processed_items, total_items):
    calls = stats["calls"]
    avg_tokens = stats["total_tokens"] / calls if calls else 0
    avg_cost = stats["cost"] / calls if calls else 0

    # Prefer estimating totals based on the known total number of LLM calls (tqdm.total)
    # when the progress bar is "per-call" (Scheme C). Fall back to per-item estimation
    # for legacy progress bars where total represents items (e.g., test_one path).
    total_calls = getattr(progress, "total", None)
    if total_calls is not None and total_calls >= calls:
        est_total_calls = total_calls
    else:
        avg_calls_per_item = calls / processed_items if processed_items else 0
        est_total_calls = avg_calls_per_item * total_items

    est_total_tokens = avg_tokens * est_total_calls
    est_total_cost = avg_cost * est_total_calls

    columns = shutil.get_terminal_size((120, 20)).columns
    postfix_items = [
        ("items", f"{processed_items}/{total_items}"),
        ("calls", f"{calls}"),
        ("avg_tok", format_tokens(avg_tokens)),
        ("tot_tok", format_tokens(stats["total_tokens"])),
        ("est_tok", format_tokens(est_total_tokens)),
        ("avg_cost", f"${avg_cost:.6f}"),
        ("tot_cost", f"${stats['cost']:.6f}"),
        ("est_cost", f"${est_total_cost:.4f}"),
    ]
    postfix_dict = dict(postfix_items)

    progress.set_postfix(postfix_dict, refresh=False)


def build_prompt_for_unknown(head, candidate_rels, is_zero, args, prompt_dict):
    # head = clean_symbol_in_rel(head)
    chain_defination = prompt_dict['chain_defination'].format(head=head)

    context = prompt_dict['unknown_relation_context'].format(head=head)

    predict = prompt_dict['unknown_relation_final_predict'].format(head=head)
    return_rules = prompt_dict['unknown_relation_return']
    return chain_defination + context, predict, return_rules


def build_confidence_with_names(rules_dict, id2relation):
    """
    Enrich confidence dict with relation names and abstract rule string.
    Used when rules come from LLM (no concrete paths available).
    """
    output = {}
    for head_rel, rules in rules_dict.items():
        head_key = str(head_rel)
        output.setdefault(head_key, [])
        for rule in rules:
            item = copy.deepcopy(rule)
            item["head_rel_name"] = id2relation[rule["head_rel"]]
            item["body_rels_names"] = [id2relation[x] for x in rule["body_rels"]]
            item["abstract_rule"] = verbalize_rule(copy.deepcopy(rule), id2relation)
            output[head_key].append(item)
    return output


def _normalize_var_constraints_for_sort(var_constraints):
    vc = var_constraints or []
    out = []
    for grp in vc:
        if isinstance(grp, (list, tuple)):
            out.append(tuple(int(x) for x in grp))
        else:
            out.append((int(grp),))
    return tuple(sorted(tuple(sorted(g)) for g in out))


def _rule_sort_key(rule):
    return (
        -float(rule.get("conf", 0) or 0),
        -float(rule.get("llm_confidence", 0) or 0),
        -int(rule.get("rule_supp", 0) or 0),
        -int(rule.get("body_supp", 0) or 0),
        tuple(int(x) for x in (rule.get("body_rels") or [])),
        _normalize_var_constraints_for_sort(rule.get("var_constraints")),
    )


def stable_sort_rules_dict(rules_dict):
    """
    Make rules_dict serialization stable:
    - sort head_rel keys ascending
    - sort each head's rule list by a deterministic key
    """
    out = {}
    for head_rel in sorted(rules_dict.keys(), key=lambda x: int(x)):
        rules = rules_dict.get(head_rel) or []
        out[int(head_rel)] = sorted(rules, key=_rule_sort_key)
    return out


def get_rule_format(head, path, kg_rules_path):
    kg_rules_dict = load_json_data(kg_rules_path)
    if kg_rules_dict is None:
        path_list = []
        # head = clean_symbol_in_rel(head)
        for p in path:
            context = f"{head}(X,Y) <-- "
            for i, r in enumerate(p.split("|")):
                # r = clean_symbol_in_rel(r)
                if i == 0:
                    first = "X"
                else:
                    first = f"Z_{i}"
                if i == len(p.split("|")) - 1:
                    last = "Y"
                else:
                    last = f"Z_{i + 1}"
                context += f"{r}({first}, {last}) & "
            context = context.strip(" & ")
            path_list.append(context)
        return path_list
    else:
        return kg_rules_dict[head]


def generate_rule(row, rdict, common_query_and_rule, kg_rules_path, model, args, relation_regex,
                  similiary_rel_dict, prompt_info_dict, prompt_info_dict_ex1, example_pool_index=None, progress_call=None):
    usage_summary = init_usage_summary()
    relation2id = rdict.rel2idx
    head_name = row["head"]
    all_paths_from_train = row["paths"]
    head_id = relation2id[head_name]

    head_formatted = head_name
    if args.is_rel_name is True:
        all_rels = list(relation2id.keys())
        candidate_rels = ", ".join(all_rels)
        head_formatted = head_name
    else:
        all_rels = list(relation2id.values())
        str_list = [str(item) for item in all_rels]
        candidate_rels = ", ".join(str_list)
        head_formatted = head_id

    # Raise an error if k=0 for zero-shot setting
    if args.k == 0 and args.is_zero:
        raise NotImplementedError(
            f"""Cannot implement for zero-shot(f=0) and generate zero(k=0) rules."""
        )
    # Build prompt excluding rules
    readable_all_path_from_train = get_rule_format(head_name, all_paths_from_train, kg_rules_path)
    
    if args.ex1_prompt is True:
        definition , role , examples_title , examples, sample_random_walk_paths_title, candidate, rule_head_title_for_user, generated_temporal_logic_rules_title = build_prompt_for_common_ex1(head_formatted, prompt_info_dict_ex1)
    else:
        definition , role , examples_title , examples, sampled_paths_title , goal , candidate, return_rules = build_prompt_for_common(head_formatted, prompt_info_dict)
    
    if args.is_zero:  # For zero-shot setting
        current_prompt = definition +"\n\n" + role + "\n\n" + examples_title + "\n" + examples + "\n\n" + sampled_paths_title + "\n" + goal + "\n\n" + candidate + "\n\n" + return_rules
        current_prompt = definition +"\n\n" + role + "\n\n" + examples_title + "\n" + examples + "\n\n" + sampled_paths_title + "\n" + goal + "\n\n" + candidate + "\n\n" + return_rules
        with safe_open(os.path.join(common_query_and_rule, f"{head_name}_zero_shot.query"), "w", encoding="utf-8") as f:
            f.write(current_prompt + "\n")
            f.close()
        if not args.dry_run:
            try:
                generated = model.generate_sentence(current_prompt, return_usage=True)
            finally:
                if progress_call is not None:
                    progress_call()
            if isinstance(generated, tuple):
                response, usage_meta = generated
            else:
                response, usage_meta = generated, None
            with safe_open(os.path.join(common_query_and_rule, f"{head_name}_zero_shot.txt"), "w", encoding="utf-8") as f:
                f.write(response + "\n")
                f.close()
            usage_summary = merge_usage_summary(
                usage_summary,
                collect_usage_from_generation(model, current_prompt, response, usage_meta),
            )
    else:  # For few-shot setting
        file_name = head_name.replace("/", "-")
        with safe_open(maybe_windows_long_path(os.path.join(common_query_and_rule, f"{file_name}.txt")), "w", encoding="utf-8") as rule_file, safe_open(
                maybe_windows_long_path(os.path.join(common_query_and_rule, f"{file_name}.query")), "w", encoding="utf-8") as query_file:
            rule_file.write(f"Rule_head: {head_name}\n")
            for i in range(args.l):
                rng_paths = get_task_rng(getattr(args, "base_seed", 0), "common_legacy", head_name, i, "paths")
                rng_truncate = get_task_rng(getattr(args, "base_seed", 0), "common_legacy", head_name, i, "truncate_paths")

                if args.select_with_confidence is True:
                    sorted_list = sorted(readable_all_path_from_train, key=lambda x: float(
                        x.split('&')[-1]), reverse=True)
                    # few_shot_samples = sorted_list[:args.f]
                    new_shot_samples = [item for item in sorted_list if float(
                        item.split('&')[-1]) > 0.01]
                    if len(new_shot_samples) >= args.f:
                        sampled_paths = new_shot_samples
                    else:
                        sampled_paths = sorted_list[:args.f]
                else:
                    sampled_paths = rng_paths.sample(
                        readable_all_path_from_train, min(args.f, len(readable_all_path_from_train))
                    )

                relation_set = set()
                for rule in sampled_paths:
                    rule_body = rule.split('<-')[-1]
                    matches = re.findall(relation_regex, rule_body)
                    for match in matches:
                        relation = match[0]
                        relation_set.update([relation])

                similiary_rel_set = set()
                for rel_name in relation_set:
                    similiary_rel_set.update(similiary_rel_dict[rel_name])

                candidate_relations = similiary_rel_set.union(relation_set)
                candidate_relations_formatted = format_candidate_relations(candidate_relations)

                # Dynamic examples (retrieval-based few-shot):
                # - Query = current sampled_paths (same ones to be inserted into the prompt)
                # - Similarity = TF-IDF cosine over relation-name tokens extracted from rule bodies
                # - Diversity = MMR (Maximal Marginal Relevance) to avoid near-duplicate examples
                # This is gated by CLI flag to keep baseline reproducibility unchanged by default.
                examples_text_for_prompt = examples
                if getattr(args, "use_dynamic_examples", False) and example_pool_index is not None:
                    if getattr(args, "base_seed", 0):
                        dyn_seed = derive_seed_64(getattr(args, "base_seed", 0), "dynex", "common_legacy", head_name, i) % (2 ** 32)
                    else:
                        dyn_seed = int.from_bytes(os.urandom(4), "little", signed=False)
                    retrieved_items = retrieve_dynamic_examples(
                        sampled_paths,
                        example_pool_index,
                        k=getattr(args, "dynamic_k", 3),
                        top_m=getattr(args, "dynamic_top_m", 20),
                        lambda_mmr=getattr(args, "dynamic_lambda_mmr", 0.7),
                        mode=args.ex1_mode,
                        random_seed=dyn_seed,
                    )
                    if args.ex1_prompt is True:
                        dynamic_examples_text = format_examples_ex1(retrieved_items)
                    else:
                        dynamic_examples_text = format_examples(retrieved_items)
                    if dynamic_examples_text:
                        examples_text_for_prompt = dynamic_examples_text

                candidate_text = candidate.format(candidate_rels=candidate_relations_formatted)

                if args.ex1_prompt is True:
                    temp_current_prompt = definition +"\n\n" + role + "\n\n"+ candidate_text + "\n\n" + examples_title + "\n" + examples_text_for_prompt + "\n\n" + rule_head_title_for_user + "\n" + sample_random_walk_paths_title
                else:
                    temp_current_prompt = definition +"\n\n" + role + "\n\n" + examples_title + "\n" + examples_text_for_prompt + "\n\n" + sampled_paths_title + "\n" + goal + "\n\n" + candidate_text + "\n\n" + return_rules

                sampled_paths_checked = check_prompt_length(
                    temp_current_prompt,
                    sampled_paths, model, rng=rng_truncate
                )

                if not sampled_paths_checked:
                    raise ValueError(
                        "few_shot_paths is empty, head:{}".format(head_name))

                if args.ex1_prompt is True:
                    prompt = temp_current_prompt + "\n" + sampled_paths_checked + generated_temporal_logic_rules_title + "\n"
                else:    
                    prompt = definition +"\n\n" + role + "\n\n" + examples_title + "\n" + examples_text_for_prompt + "\n\n" + sampled_paths_title + "\n" + sampled_paths_checked +"\n"+ goal + "\n\n" + candidate_text + "\n\n" + return_rules
                
                
                query_file.write(f"Sample {i + 1} time: \n")
                query_file.write(prompt + "\n")
                if not args.dry_run:
                    try:
                        generated = model.generate_sentence(prompt, return_usage=True)
                    finally:
                        if progress_call is not None:
                            progress_call()
                    if isinstance(generated, tuple):
                        response, usage_meta = generated
                    else:
                        response, usage_meta = generated, None
                    if response is not None:
                        # tqdm.write("Response: \n{}".format(response))
                        rule_file.write(f"Sample {i + 1} time: \n")
                        rule_file.write(response + "\n")
                        usage_summary = merge_usage_summary(
                            usage_summary,
                            collect_usage_from_generation(model, prompt, response, usage_meta),
                        )
                    else:
                        with safe_open(os.path.join(common_query_and_rule, f"fail_{file_name}.txt"), "w", encoding="utf-8") as fail_rule_file:
                            fail_rule_file.write(prompt + "\n")
                        break

    return usage_summary


def generate_rule_for_zero(head, rdict, zero_query_and_rule, model, args, prompt_info_zero_dict, progress_call=None):
    usage_summary = init_usage_summary()
    relation2id = rdict.rel2idx
    all_rels_name = list(relation2id.keys())

    definition, role , examples_title , examples , goal , candidate, return_rules = build_prompt_for_zero(head, prompt_info_zero_dict)
    candidate = candidate.format(candidate_rels=all_rels_name)
    current_prompt = definition + "\n\n" + role + "\n\n" + examples_title + "\n" + examples + "\n\n" + goal + "\n\n" + candidate + "\n\n" + return_rules

    # 定义文件路径
    query_file_path = maybe_windows_long_path(os.path.join(zero_query_and_rule, f"{head}.query"))
    txt_file_path = maybe_windows_long_path(os.path.join(zero_query_and_rule, f"{head}.txt"))
    fail_file_path = maybe_windows_long_path(os.path.join(zero_query_and_rule, f"fail_{head}.txt"))

    try:
        with safe_open(query_file_path, "w", encoding="utf-8") as fout_zero_query, safe_open(txt_file_path, "w", encoding="utf-8") as fout_zero_txt:
            for i in range(args.l):
                entry = f"Sample {i + 1} time:\n"
                fout_zero_query.write(entry + current_prompt + "\n")
                try:
                    generated = model.generate_sentence(current_prompt, return_usage=True)
                finally:
                    if progress_call is not None:
                        progress_call()
                if isinstance(generated, tuple):
                    response, usage_meta = generated
                else:
                    response, usage_meta = generated, None
                if response:
                    fout_zero_txt.write(entry + response + "\n")
                    usage_summary = merge_usage_summary(
                        usage_summary,
                        collect_usage_from_generation(model, current_prompt, response, usage_meta),
                    )
                else:
                    raise ValueError("Failed to generate response.")
    except ValueError as e:
        with safe_open(fail_file_path, "w", encoding="utf-8") as fail_rule_file:
            fail_rule_file.write(current_prompt + "\n")
        print(e)  # Optional: Handle the exception as needed

    return usage_summary


class _OrderedSampleWriter:
    def __init__(self, query_path, txt_path, fail_path=None, include_rule_head=False, rule_head=None):
        self._lock = threading.Lock()
        self._next_idx = 0
        self._pending = {}
        self.query_path = query_path
        self.txt_path = txt_path
        self.fail_path = fail_path
        self._include_rule_head = include_rule_head
        self._rule_head = rule_head

    def init_files(self):
        with self._lock:
            with safe_open(self.query_path, "w", encoding="utf-8"):
                pass
            with safe_open(self.txt_path, "w", encoding="utf-8") as tf:
                if self._include_rule_head and self._rule_head is not None:
                    tf.write(f"Rule_head: {self._rule_head}\n")

            if self.fail_path is not None and os.path.exists(self.fail_path):
                try:
                    os.remove(self.fail_path)
                except Exception:
                    pass

            self._next_idx = 0
            self._pending.clear()

    def submit(self, sample_idx, prompt, response):
        with self._lock:
            self._pending[sample_idx] = (prompt, response)
            self._flush_locked()

    def _flush_locked(self):
        while self._next_idx in self._pending:
            prompt, response = self._pending.pop(self._next_idx)
            sample_no = self._next_idx + 1

            with safe_open(self.query_path, "a", encoding="utf-8") as qf:
                qf.write(f"Sample {sample_no} time: \n")
                qf.write(prompt + "\n")

            if response is not None:
                with safe_open(self.txt_path, "a", encoding="utf-8") as tf:
                    tf.write(f"Sample {sample_no} time: \n")
                    tf.write(response + "\n")
            elif self.fail_path is not None:
                with safe_open(self.fail_path, "a", encoding="utf-8") as ff:
                    ff.write(f"Sample {sample_no} time: \n")
                    ff.write(prompt + "\n")

            self._next_idx += 1


def _run_common_call_task(task, head_states, model, args, relation_regex,
                          similiary_rel_dict, example_pool_index, profiles=None):
    head_key, sample_idx = task
    state = head_states[head_key]

    prompt = ""
    response = None
    usage_meta = None
    start_time = time.time()
    try:
        if state.get("is_zero_shot", False):
            prompt = state["prompt"]
            if not args.dry_run:
                generated = model.generate_sentence(prompt, return_usage=True)
                if isinstance(generated, tuple):
                    response, usage_meta = generated
                else:
                    response, usage_meta = generated, None
        else:
            rng_paths = get_task_rng(getattr(args, "base_seed", 0), "common", head_key, sample_idx, "paths")
            rng_truncate = get_task_rng(getattr(args, "base_seed", 0), "common", head_key, sample_idx, "truncate_paths")
            if args.select_with_confidence is True:
                sorted_list = sorted(
                    state["readable_all_path_from_train"],
                    key=lambda x: float(x.split('&')[-1]),
                    reverse=True,
                )
                new_shot_samples = [item for item in sorted_list if float(item.split('&')[-1]) > 0.01]
                if len(new_shot_samples) >= args.f:
                    sampled_paths = new_shot_samples
                else:
                    sampled_paths = sorted_list[:args.f]
            else:
                sampled_paths = rng_paths.sample(
                    state["readable_all_path_from_train"],
                    min(args.f, len(state["readable_all_path_from_train"])),
                )

            relation_set = set()
            for rule in sampled_paths:
                rule_body = rule.split('<-')[-1]
                matches = re.findall(relation_regex, rule_body)
                for match in matches:
                    relation = match[0]
                    relation_set.update([relation])

            similiary_rel_set = set()
            for rel_name in relation_set:
                similiary_rel_set.update(similiary_rel_dict[rel_name])

            candidate_relations = similiary_rel_set.union(relation_set)
            candidate_relations_formatted = format_candidate_relations(candidate_relations)

            examples_text_for_prompt = state["examples"]
            if getattr(args, "use_dynamic_examples", False) and example_pool_index is not None:
                if getattr(args, "base_seed", 0):
                    dyn_seed = derive_seed_64(getattr(args, "base_seed", 0), "dynex", "common", head_key, sample_idx) % (2 ** 32)
                else:
                    dyn_seed = int.from_bytes(os.urandom(4), "little", signed=False)
                retrieved_items = retrieve_dynamic_examples(
                    sampled_paths,
                    example_pool_index,
                    k=getattr(args, "dynamic_k", 3),
                    top_m=getattr(args, "dynamic_top_m", 20),
                    lambda_mmr=getattr(args, "dynamic_lambda_mmr", 0.7),
                    mode=args.ex1_mode,
                    random_seed=dyn_seed,
                )
                if args.ex1_prompt is True:
                    dynamic_examples_text = format_examples_ex1(retrieved_items)
                else:
                    dynamic_examples_text = format_examples(retrieved_items)              
                if dynamic_examples_text:
                    examples_text_for_prompt = dynamic_examples_text

            candidate_text = state["candidate_template"].format(candidate_rels=candidate_relations_formatted)

            head_name = state.get("head_name", head_key)
            profile_block = ""
            if getattr(args, "use_semantic_profile", False) and profiles:
                profile_block = build_profile_block(
                    head_name,
                    profiles,
                    use_country=getattr(args, "profile_use_country_pairs", True),
                    use_time=getattr(args, "profile_use_time_stats", True),
                    use_events=getattr(args, "profile_use_repr_events", True),
                    topk_pairs=getattr(args, "profile_topk_pairs", 10),
                    events_k=getattr(args, "profile_events_k", 3),
                )

            candidate_with_profile = candidate_text
            if profile_block:
                candidate_with_profile = candidate_text + "\n\n" + profile_block

            if state["ex1_prompt"] is True:
                temp_current_prompt = (
                    state["definition"]
                    + "\n\n" + state["role"]
                    + "\n\n" + candidate_with_profile
                    + "\n\n" + state["examples_title"] + "\n" + examples_text_for_prompt
                    + "\n\n" + state["rule_head_title_for_user"] + "\n" + state["sample_random_walk_paths_title"]
                )
            else:
                temp_current_prompt = (
                    state["definition"]
                    + "\n\n" + state["role"]
                    + "\n\n" + state["examples_title"] + "\n" + examples_text_for_prompt
                    + "\n\n" + state["sampled_paths_title"] + "\n" + state["goal"]
                    + "\n\n" + candidate_with_profile
                    + "\n\n" + state["return_rules"]
                )

            sampled_paths_checked = check_prompt_length(temp_current_prompt, sampled_paths, model, rng=rng_truncate)
            if not sampled_paths_checked:
                raise ValueError(f"few_shot_paths is empty, head:{state['head_name']}")

            if state["ex1_prompt"] is True:
                prompt = temp_current_prompt + "\n" + sampled_paths_checked + state["generated_temporal_logic_rules_title"] + "\n"
            else:
                prompt = (
                    state["definition"]
                    + "\n\n" + state["role"]
                    + "\n\n" + state["examples_title"] + "\n" + examples_text_for_prompt
                    + "\n\n" + state["sampled_paths_title"] + "\n" + sampled_paths_checked + "\n" + state["goal"]
                    + "\n\n" + candidate_with_profile
                    + "\n\n" + state["return_rules"]
                )

            if not args.dry_run:
                generated = model.generate_sentence(prompt, return_usage=True)
                if isinstance(generated, tuple):
                    response, usage_meta = generated
                else:
                    response, usage_meta = generated, None
    except Exception:
        prompt = prompt or traceback.format_exc()
        response = None
        usage_meta = None

    elapsed = time.time() - start_time
    if usage_meta is None:
        usage_meta = {"elapsed": elapsed, "cost": 0.0}
    elif isinstance(usage_meta, dict) and usage_meta.get("elapsed") is None:
        usage_meta = dict(usage_meta)
        usage_meta["elapsed"] = elapsed

    usage = collect_usage_from_generation(model, prompt, response, usage_meta)
    state["writer"].submit(sample_idx, prompt, response)
    return head_key, usage


def _run_zero_call_task(task, head_states, model, args):
    head_key, sample_idx = task
    state = head_states[head_key]

    prompt = state["prompt"]
    response = None
    usage_meta = None
    start_time = time.time()
    try:
        if not args.dry_run:
            generated = model.generate_sentence(prompt, return_usage=True)
            if isinstance(generated, tuple):
                response, usage_meta = generated
            else:
                response, usage_meta = generated, None
    except Exception:
        response = None
        usage_meta = None

    elapsed = time.time() - start_time
    if usage_meta is None:
        usage_meta = {"elapsed": elapsed, "cost": 0.0}
    elif isinstance(usage_meta, dict) and usage_meta.get("elapsed") is None:
        usage_meta = dict(usage_meta)
        usage_meta["elapsed"] = elapsed

    usage = collect_usage_from_generation(model, prompt, response, usage_meta)
    state["writer"].submit(sample_idx, prompt, response)
    return head_key, usage


def _run_iteration_call_task(task, head_states, model, args, relation_regex, similiary_rel_dict,
                             sampled_paths_dict_from_train, valid_paths_dict, prompt_dict_for_low, prompt_dict_for_high):
    head_key, sample_idx = task
    state = head_states[head_key]

    prompt = ""
    response = None
    usage_meta = None
    start_time = time.time()
    try:
        rng_expand = get_task_rng(getattr(args, "base_seed", 0), "iteration", head_key, sample_idx, "expand_rels")
        rng_valid_paths = get_task_rng(getattr(args, "base_seed", 0), "iteration", head_key, sample_idx, "valid_paths")
        rng_low_rules = get_task_rng(getattr(args, "base_seed", 0), "iteration", head_key, sample_idx, "low_rules")
        rng_truncate_candidates = get_task_rng(getattr(args, "base_seed", 0), "iteration", head_key, sample_idx, "truncate_candidates")
        if state["based_rule_type"] == "high":
            fixed_context, predict, return_rules = build_prompt_based_high(
                state["head_formate"], state["candidate_rels_all"], args.is_zero, args, prompt_dict_for_high
            )
            prompt = fixed_context + "\n\n" + predict + "\n\n" + return_rules
        else:
            sampled_paths_for_one_head_from_train = sampled_paths_dict_from_train.get(state["selected_head_name"])
            if sampled_paths_for_one_head_from_train is not None:
                candicate_relations = extract_and_expand_relations(
                    args, sampled_paths_for_one_head_from_train, similiary_rel_dict, relation_regex, rng=rng_expand
                )
            else:
                candicate_relations = set(state["all_rels"])

            valid_paths_for_one_head = valid_paths_dict.get(state["selected_head_name"], []) or []
            valid_paths_for_one_head_sampled = rng_valid_paths.sample(
                valid_paths_for_one_head, min(20, len(valid_paths_for_one_head))
            )

            selected_conf_rules_sampled = rng_low_rules.sample(
                state["selected_conf_rules"], min(20, len(state["selected_conf_rules"]))
            )
            selected_conf_rules_sampled_formatted_string = ''.join(selected_conf_rules_sampled)
            valid_paths_for_one_head_sampled_formatted_string = ''.join(valid_paths_for_one_head_sampled)

            temp_current_prompt = (
                state["definition"]
                + "\n\n" + state["role"]
                + "\n\n" + state["examples_title"] + "\n" + state["examples"]
                + "\n\n" + state["low_quality_rules_title"] + "\n" + selected_conf_rules_sampled_formatted_string
                + "\n" + state["sample_rules_title"] + "\n" + valid_paths_for_one_head_sampled_formatted_string
                + "\n" + state["goal"]
            )

            ordered_candidates = [r for r in state.get("all_rels", []) if r in candicate_relations]
            if not ordered_candidates:
                ordered_candidates = sorted(candicate_relations)
            formatted_candidate_relations_string = iteration_check_prompt_length(
                temp_current_prompt,
                ordered_candidates,
                state["candidate_template"] + "\n\n" + state["return_rules"],
                model,
                rng=rng_truncate_candidates,
            )
            candidate_text = state["candidate_template"].format(candidate_rels=formatted_candidate_relations_string)
            prompt = temp_current_prompt + "\n\n" + candidate_text + "\n\n" + state["return_rules"]

        if not args.dry_run:
            generated = model.generate_sentence(prompt, return_usage=True)
            if isinstance(generated, tuple):
                response, usage_meta = generated
            else:
                response, usage_meta = generated, None
    except Exception:
        prompt = prompt or traceback.format_exc()
        response = None
        usage_meta = None

    elapsed = time.time() - start_time
    if usage_meta is None:
        usage_meta = {"elapsed": elapsed, "cost": 0.0}
    elif isinstance(usage_meta, dict) and usage_meta.get("elapsed") is None:
        usage_meta = dict(usage_meta)
        usage_meta["elapsed"] = elapsed

    usage = collect_usage_from_generation(model, prompt, response, usage_meta)
    state["writer"].submit(sample_idx, prompt, response)
    return head_key, usage


def extract_and_expand_relations(args, path_content_list, similiary_rel_dict, relation_regex, rng=None):
    """
    从提供的规则样本中随机抽取一定数量的规则，并扩展这些规则中的关系集合。

    :param args: 命名空间，包含参数f，表示要抽取的规则数量。
    :param path_content_list: 包含规则的列表。
    :param similiary_rel_dict: 一个字典，键为关系名，值为与该关系相似的关系集合。
    :param relation_regex: 用于从规则中提取关系的正则表达式。
    :return: 包含原始关系和相似关系的集合。
    """
    # 随机抽取样本
    if rng is None:
        rng = random
    few_shot_samples = rng.sample(path_content_list, min(args.f, len(path_content_list)))

    # 从抽取的样本中提取关系
    relation_set = set()
    for rule in few_shot_samples:
        rule_body = rule.split('<-')[-1]
        matches = re.findall(relation_regex, rule_body)
        for match in matches:
            relation = match[0]
            relation_set.update([relation])

    # 扩展找到的关系集合，包括相似的关系
    similiary_rel_set = set()
    for rel_name in relation_set:
        similiary_rel_set.update(similiary_rel_dict[rel_name])

    # 合并原始关系集和相似关系集
    condicate = similiary_rel_set.union(relation_set)

    return condicate


def generate_rule_for_iteration_by_multi_thread(selected_conf_rule_for_one_head, rdict, rule_path, kg_rules_path, model, args,
                                                relation_regex,
                                                similiary_rel_dict, kg_rules_path_with_valid, prompt_dict_for_low, prompt_dict_for_high):
    usage_summary = init_usage_summary()
    relation2id = rdict.rel2idx
    selected_conf_rules_head_name = selected_conf_rule_for_one_head["head"]
    selected_conf_rules = selected_conf_rule_for_one_head["rules"]

    selected_conf_rules_head_id = relation2id[selected_conf_rules_head_name]

    valid_paths_dict = load_json_data(kg_rules_path_with_valid)

    if args.is_rel_name is True:
        all_rels = list(relation2id.keys())
        candidate_rels_all = ", ".join(all_rels)
        head_formate = selected_conf_rules_head_name
    else:
        all_rels = list(relation2id.values())
        str_list = [str(item) for item in all_rels]
        candidate_rels_all = ", ".join(str_list)
        head_formate = selected_conf_rules_head_id

    # Raise an error if k=0 for zero-shot setting
    if args.k == 0 and args.is_zero:
        raise NotImplementedError(
            f"""Cannot implement for zero-shot(f=0) and generate zero(k=0) rules."""
        )

    if args.based_rule_type == 'low':
        # Build prompt excluding rules
        definition , role , examples_title , examples, low_quality_rules_title, sample_rules_title , goal , candidate, return_rules = build_prompt_based_low(head_formate, candidate_rels_all, args.is_zero, args, prompt_dict_for_low)
    else:
        fixed_context, goal, return_rules = build_prompt_based_high(
            head_formate, candidate_rels_all, args.is_zero, args, prompt_dict_for_high
        )

    sampled_paths_dict_from_train = load_json_data(kg_rules_path)
    sampled_paths_for_one_head_from_train = sampled_paths_dict_from_train.get(selected_conf_rules_head_name, None)
    file_name = selected_conf_rules_head_name.replace("/", "-")
    with safe_open(os.path.join(rule_path, f"{file_name}.txt"), "w", encoding="utf-8") as rule_file, safe_open(
            os.path.join(rule_path, f"{file_name}.query"), "w", encoding="utf-8") as query_file:
        rule_file.write(f"Rule_head: {selected_conf_rules_head_name}\n")
        candidate_template = candidate
        for i in range(args.second):
            rng_expand = get_task_rng(getattr(args, "base_seed", 0), "iteration_legacy", selected_conf_rules_head_name, i, "expand_rels")
            rng_valid_paths = get_task_rng(getattr(args, "base_seed", 0), "iteration_legacy", selected_conf_rules_head_name, i, "valid_paths")
            rng_low_rules = get_task_rng(getattr(args, "base_seed", 0), "iteration_legacy", selected_conf_rules_head_name, i, "low_rules")
            rng_truncate_candidates = get_task_rng(getattr(args, "base_seed", 0), "iteration_legacy", selected_conf_rules_head_name, i, "truncate_candidates")

            if sampled_paths_for_one_head_from_train is not None:
                candicate_relations = extract_and_expand_relations(
                    args, sampled_paths_for_one_head_from_train, similiary_rel_dict, relation_regex, rng=rng_expand)
            else:
                candicate_relations = set(all_rels)

            valid_paths_for_one_head = valid_paths_dict.get(selected_conf_rules_head_name, []) or []
            valid_paths_for_one_head_sampled = rng_valid_paths.sample(
                valid_paths_for_one_head,  min(20, len(valid_paths_for_one_head)))

            selected_conf_rules_sampled = rng_low_rules.sample(selected_conf_rules, min(20, len(selected_conf_rules)))
            selected_conf_rules_sampled_formatted_string = ''.join(selected_conf_rules_sampled)

            valid_paths_for_one_head_sampled_formatted_string = ''.join(valid_paths_for_one_head_sampled)

            temp_current_prompt = definition +"\n\n" + role + "\n\n" + examples_title + "\n" + examples + "\n\n" + low_quality_rules_title + "\n" + \
                selected_conf_rules_sampled_formatted_string +"\n"+ sample_rules_title + "\n" + valid_paths_for_one_head_sampled_formatted_string + "\n" + goal

            ordered_candidates = [r for r in all_rels if r in candicate_relations]
            if not ordered_candidates:
                ordered_candidates = sorted(candicate_relations)
            formatted_candidate_relations_string = iteration_check_prompt_length(
                temp_current_prompt,
                ordered_candidates, candidate_template + "\n\n" + return_rules, model, rng=rng_truncate_candidates
            )

            candidate_text = candidate_template.format(candidate_rels=formatted_candidate_relations_string)
            
            prompt = temp_current_prompt + "\n\n" + candidate_text + "\n\n" + return_rules
            
            query_file.write(f"Sample {i + 1} time: \n")
            query_file.write(prompt + "\n")
            if not args.dry_run:
                generated = model.generate_sentence(prompt, return_usage=True)
                if isinstance(generated, tuple):
                    response, usage_meta = generated
                else:
                    response, usage_meta = generated, None
                if response is not None:
                    # tqdm.write("Response: \n{}".format(response))
                    rule_file.write(f"Sample {i + 1} time: \n")
                    rule_file.write(response + "\n")
                    usage_summary = merge_usage_summary(
                        usage_summary,
                        collect_usage_from_generation(model, prompt, response, usage_meta),
                    )
                else:
                    with safe_open(os.path.join(rule_path, f"fail_{file_name}.txt"), "w", encoding="utf-8") as fail_rule_file:
                        fail_rule_file.write(prompt + "\n")

    return usage_summary

def copy_files(source_dir, destination_dir, file_extension):
    # 创建目标文件夹
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # 遍历源文件夹中的文件
    for filename in sorted(os.listdir(source_dir), key=str.lower):
        # 检查文件类型是否符合要求
        if filename.endswith(file_extension):
            source_file = maybe_windows_long_path(os.path.join(source_dir, filename))
            destination_file = maybe_windows_long_path(os.path.join(destination_dir, filename))
            # 复制文件
            shutil.copyfile(source_file, destination_file)


def process_rules_files(input_dir, output_dir, rdict, relation_regex, error_file_path):
    sum = 0
    with safe_open(error_file_path, 'w', encoding="utf-8") as f_error_out:
        for input_filepath in sorted(
            glob.glob(maybe_windows_long_path(os.path.join(input_dir, "*.txt"))),
            key=str.lower,
        ):
            file_name = os.path.basename(input_filepath)
            if file_name.startswith('fail'):
                continue
            else:
                with safe_open(input_filepath, 'r', encoding="utf-8") as fin, safe_open(os.path.join(output_dir, file_name), 'w', encoding="utf-8") as fout:
                    rules = fin.readlines()
                    for idx, rule in enumerate(rules):
                        is_save = True
                        if rule.startswith('Rule_head:'):
                            continue
                        elif rule.startswith('Sample'):
                            continue
                        else:
                            rule_by_name = ""
                            temp_rule = re.sub(r'\s*<-\s*', '&', rule)
                            regrex_list = re.split(r'\s*&\s*|\t', temp_rule)
                            confidence = regrex_list[-1].strip()
                            for id, regrex in enumerate(regrex_list[:-1]):
                                match = re.search(relation_regex, regrex)
                                if match:
                                    if match[1].strip().isdigit():
                                        rel_id = int(match[1].strip())
                                        if rel_id not in list(rdict.idx2rel):
                                            print(
                                                f"Error relation id:{rel_id}, rule:{rule}")
                                            f_error_out.write(
                                                f"Error relation id:{rel_id}, rule:{rule}")
                                            sum = sum + 1
                                            is_save = False
                                            break

                                        relation_name = rdict.idx2rel[rel_id]
                                        subject = match[2].strip()
                                        object = match[3].strip()
                                        timestamp = match[4].strip()
                                        regrex_name = f"{relation_name}({subject},{object},{timestamp})"
                                        if id == 0:
                                            regrex_name += '<-'
                                        else:
                                            regrex_name += '&'
                                        rule_by_name += regrex_name
                                    else:
                                        print(
                                            f"Error relation id:{match[1].strip()}, rule:{rule}")
                                        f_error_out.write(
                                            f"Error relation id:{match[1].strip()}, rule:{rule}")
                                        sum = sum + 1
                                        is_save = False
                                        break

                                else:
                                    print(f"Error rule:{rule}, rule:{rule}")
                                    f_error_out.write(
                                        f"Error rule:{rule}, rule:{rule}")
                                    sum = sum + 1
                                    is_save = False
                                    break
                            if is_save:
                                rule_by_name += confidence
                                fout.write(rule_by_name + '\n')
        f_error_out.write(f"The number of error during id maps name is:{sum}")


def get_topk_similiary_rel(topk, similary_matrix, transformers_id2rel, transformers_rel2id):
    # 计算每一行中数值最大的前 topk 个元素的索引
    topk = -topk
    top_k_indices = np.argsort(similary_matrix, axis=1)[:, topk:]
    similiary_rel_dict = {}
    for idx, similary_rels in enumerate(top_k_indices):
        rel_name = transformers_id2rel[str(idx)]
        similary_rel_name = [
            transformers_id2rel[str(i)] for i in similary_rels]
        similiary_rel_dict[rel_name] = similary_rel_name

    return similiary_rel_dict


def get_low_conf(low_conf_file_path, relation_regex, rdict):
    rule_dict = {}
    with safe_open(low_conf_file_path, 'r', encoding="utf-8") as fin_low:
        rules = fin_low.readlines()
        for rule in rules:
            if 'index' in rule:
                continue
            regrex_list = rule.split('<-')
            match = re.search(relation_regex, regrex_list[0])
            if match:
                head = match[1].strip()
                if head not in list(rdict.rel2idx.keys()):
                    raise ValueError(f"Not exist relation:{head}")

                if head not in rule_dict:
                    rule_dict[head] = []
                rule_dict[head].append(rule)

    rule_list = []
    for key, value in rule_dict.items():
        rule_list.append({'head': key, 'rules': value})

    return rule_list


def get_high_conf(high_conf_file_path, relation_regex, rdict):
    rule_dict = {}
    with safe_open(high_conf_file_path, 'r', encoding="utf-8") as fin_low:
        rules = fin_low.readlines()
        for rule in rules:
            if 'index' in rule:
                continue
            regrex_list = rule.split('<-')
            match = re.search(relation_regex, regrex_list[0])
            if match:
                head = match[1].strip()
                if head not in list(rdict.rel2idx.keys()):
                    raise ValueError(f"Not exist relation:{head}")

                if head not in rule_dict:
                    rule_dict[head] = []
                rule_dict[head].append(rule)

    rule_list = []
    for key, value in rule_dict.items():
        rule_list.append({'head': key, 'rules': value})

    return rule_list


def analysis_data(filter_folder, kg_rules_path):
    with safe_open(os.path.join(filter_folder, 'hight_conf.txt'), 'r', encoding="utf-8") as fin_hight, safe_open(
            os.path.join(filter_folder, 'low_conf.txt'), 'r', encoding="utf-8") as fin_low:
        hight_rule_set = set()
        rules = fin_hight.readlines()
        for rule in rules:
            if "index" in rule:
                continue
            hight_rule_set.update([rule.strip()])

        low_rule_set = set()
        rules = fin_low.readlines()
        for rule in rules:
            if "index" in rule:
                continue
            low_rule_set.update([rule.strip()])

    rules_dict = load_json_data(kg_rules_path)

    all_rules = [item.strip() for sublist in rules_dict.values()
                 for item in sublist]
    all_rules_set = set(all_rules)

    with safe_open(os.path.join(filter_folder, 'statistic.txt'), 'w', encoding="utf-8") as fout_state:
        fout_state.write(f'valid_high:{len(hight_rule_set-all_rules_set)}\n')
        fout_state.write(f'valid_low:{len(low_rule_set-all_rules_set)}\n')


def load_data_and_paths(args):
    data_path = maybe_windows_long_path(os.path.join(args.data_path, args.dataset))
    dataset = Dataset(data_root=data_path, inv=True)

    sampled_path_with_valid_dir = maybe_windows_long_path(os.path.join(args.results_root_path, args.bat_file_name, 
        args.sampled_paths, args.dataset + '_valid'))
    sampled_path_dir = maybe_windows_long_path(os.path.join(args.results_root_path, args.bat_file_name, args.sampled_paths, args.dataset))
    sampled_path_dict_only_relations = read_paths(maybe_windows_long_path(os.path.join(
        sampled_path_dir, "closed_rel_paths.jsonl")))
    prompt_path = maybe_windows_long_path(os.path.join(args.prompt_paths, 'common.json'))
    prompt_path_ex1 = maybe_windows_long_path(os.path.join(args.prompt_paths, 'common_ex1.json'))
    prompt_path_for_zero = maybe_windows_long_path(os.path.join(args.prompt_paths, 'zero.json'))
    prompt_path_for_low = maybe_windows_long_path(os.path.join(args.prompt_paths, 'low.json'))
    prompt_path_for_high = maybe_windows_long_path(os.path.join(args.prompt_paths, 'high.json'))

    return dataset, sampled_path_dict_only_relations, sampled_path_with_valid_dir, sampled_path_dir, prompt_path, prompt_path_ex1, prompt_path_for_zero, prompt_path_for_low, prompt_path_for_high


def prepare_rule_heads(dataset, sampled_path_dict_only_relations):
    rule_heads_with_path_set = {rule['head'] for rule in sampled_path_dict_only_relations}
    rule_heads_without_path_set = set(dataset.rdict.rel2idx.keys()) - rule_heads_with_path_set
    rule_heads_with_path = sorted(rule_heads_with_path_set)
    rule_heads_without_path = sorted(rule_heads_without_path_set)
    return rule_heads_with_path, rule_heads_without_path


def determine_kg_rules_path(args, sampled_path_dir):
    if args.is_rel_name:
        return maybe_windows_long_path(os.path.join(sampled_path_dir, "rules_name.json"))
    else:
        return maybe_windows_long_path(os.path.join(sampled_path_dir, "rules_id.json"))


def load_configuration(dataset, sampled_path_dir, args):
    constant_config = load_json_data(maybe_windows_long_path(os.path.join("Config", "constant.json")))
    relation_regex = constant_config['relation_regex'][args.dataset]

    rdict = dataset.get_relation_dict()
    similarity_matrix = np.load(maybe_windows_long_path(os.path.join(sampled_path_dir, "matrix.npy")))
    transformers_id2rel = load_json_data(maybe_windows_long_path(os.path.join(
        sampled_path_dir, "transfomers_id2rel.json")))
    transformers_rel2id = load_json_data(maybe_windows_long_path(os.path.join(
        sampled_path_dir, "transfomers_rel2id.json")))

    similar_rel_dict = get_topk_similiary_rel(
        args.topk, similarity_matrix, transformers_id2rel, transformers_rel2id)

    return rdict, relation_regex, similar_rel_dict

                            
def init_rule_path(args):
    args.results_root_path = args.results_root_path.strip('"')
    
    if args.is_generate_pathes_for_train_dataset == True:
        args.iteration_reasonsing_results_root_path = maybe_windows_long_path(os.path.join(args.results_root_path, args.bat_file_name, 'gen_rules_iteration_for_example_pool'))
    else:
        args.iteration_reasonsing_results_root_path = maybe_windows_long_path(os.path.join(args.results_root_path, args.bat_file_name, 'gen_rules_iteration'))
    
    root_path = maybe_windows_long_path(args.iteration_reasonsing_results_root_path)
    
    if os.path.exists(root_path):
        shutil.rmtree(root_path, ignore_errors=False)
    os.makedirs(root_path, exist_ok=True)    
        
    return


def create_directories(args):
    init_rule_path(args)
    paths = {}
    dataset_root = maybe_windows_long_path(os.path.join(args.iteration_reasonsing_results_root_path, args.dataset))
    prefix_name = f"{args.prefix}{args.model_name}-top-{args.k}-f-{args.f}-l-{args.l}"

    paths["dataset_root"] = dataset_root
    paths["query_and_rule_home"] = maybe_windows_long_path(os.path.join(dataset_root, prefix_name))
    
    paths["common_query_and_rule"] = maybe_windows_long_path(os.path.join(paths["query_and_rule_home"], "common"))
    paths["zero_query_and_rule"] = maybe_windows_long_path(os.path.join(paths["query_and_rule_home"], "zero"))

    paths["iteration_list"] = [maybe_windows_long_path(os.path.join(dataset_root, f"iteration_{i}")) for i in range(args.num_iter + 1)]

    paths["clean_root"] = maybe_windows_long_path(os.path.join(args.iteration_reasonsing_results_root_path, args.dataset, "clean"))    
    paths["cleaned_statistics_list"] = [maybe_windows_long_path(os.path.join(paths["clean_root"], f"{i}_cleaned_statistics")) for i in range(args.num_iter + 1)]
    paths["cleaned_rules_list"] = [maybe_windows_long_path(os.path.join(paths["clean_root"], f"{i}_cleaned_rules")) for i in range(args.num_iter + 1)]

    paths["filter_train"] = maybe_windows_long_path(os.path.join(args.iteration_reasonsing_results_root_path, args.dataset, "filter", "train"))
    paths["filter_eva"] = maybe_windows_long_path(os.path.join(args.iteration_reasonsing_results_root_path, args.dataset, "filter", "eva"))
    paths["filter_train_eva"] = maybe_windows_long_path(os.path.join(args.iteration_reasonsing_results_root_path, args.dataset, "filter", "train_eva"))

    paths["evaluation_train"] = maybe_windows_long_path(os.path.join(args.iteration_reasonsing_results_root_path, args.dataset, "evaluation", "train"))
    paths["evaluation_eva"] = maybe_windows_long_path(os.path.join(args.iteration_reasonsing_results_root_path, args.dataset, "evaluation", "eva"))
    paths["evaluation_train_eva"] = maybe_windows_long_path(os.path.join(args.iteration_reasonsing_results_root_path, args.dataset, "evaluation", "train_eva"))
    
    paths["final_summary"] = maybe_windows_long_path(os.path.join(args.iteration_reasonsing_results_root_path, args.dataset, "final_summary"))

    base_dirs = [
        args.iteration_reasonsing_results_root_path,
        dataset_root,
        paths["query_and_rule_home"],
        paths["common_query_and_rule"],
        paths["zero_query_and_rule"],
        paths["clean_root"],
        paths["filter_train"],
        paths["filter_eva"],
        paths["filter_train_eva"],
        paths["evaluation_train"],
        paths["evaluation_eva"],
        paths["evaluation_train_eva"],
        paths["final_summary"],
    ]
    for d in base_dirs:
        os.makedirs(d, exist_ok=True)
        clear_folder(d)
        
    for d in paths["iteration_list"]:
        os.makedirs(d, exist_ok=True)
        clear_folder(d)

    for d in paths["cleaned_statistics_list"]:
        os.makedirs(d, exist_ok=True)
        clear_folder(d)
        
    for d in paths["cleaned_rules_list"]:
        os.makedirs(d, exist_ok=True)
        clear_folder(d)
    return paths



def main(args, LLM):
    paths = create_directories(args)
    dataset, sampled_path_dict_only_relations, sampled_path_with_valid_dir, sampled_path_dir, prompt_path, prompt_path_ex1, prompt_path_for_zero, prompt_path_for_low, prompt_path_for_high = load_data_and_paths(
        args)

    # Added for subset prompt-eval (2026-01-20)
    # If a subset of relations is selected for prompt iteration, only generate rules for them.
    selected_relations_path = maybe_windows_long_path(
        os.path.join(args.results_root_path, args.bat_file_name, "sampled_path", args.dataset, "selected_relations.json")
    )
    if args.selected_relations is True and os.path.exists(selected_relations_path):
        selected_payload = load_json_data(selected_relations_path) or {}
        selected_head_rel_names = selected_payload.get("selected_head_rel_names", []) or []
        if not isinstance(selected_head_rel_names, list):
            raise ValueError(
                f"`selected_head_rel_names` must be a list in {selected_relations_path}"
            )

        selected_set = set(selected_head_rel_names)
        sampled_path_dict_only_relations = [
            row for row in sampled_path_dict_only_relations
            if row.get("head") in selected_set
        ]
        heads_in_sampled_path = {row.get("head") for row in sampled_path_dict_only_relations}
        heads_in_sampled_path.discard(None)

        # Keep stable order from the json list (do NOT rely on set order).
        rule_heads_without_path = [
            h for h in selected_head_rel_names if h not in heads_in_sampled_path
        ]
        rule_heads_with_path = set(heads_in_sampled_path)

        print(
            f"[Subset] Use selected_relations.json: {selected_relations_path}\n"
            f"[Subset] sampled_path heads kept: {len(rule_heads_with_path)} | zero-path heads: {len(rule_heads_without_path)}"
        )
    else:
        rule_heads_with_path, rule_heads_without_path = prepare_rule_heads(
            dataset, sampled_path_dict_only_relations)
    kg_rules_path = determine_kg_rules_path(args, sampled_path_dir)
    rdict, relation_regex, similar_rel_dict = load_configuration(
        dataset, sampled_path_dir, args)

    prompt_info_dict = load_json_data(prompt_path)
    prompt_info_dict_ex1 = load_json_data(prompt_path_ex1)
    prompt_info_zero_dict = load_json_data(prompt_path_for_zero)
    prompt_dict_for_low = load_json_data(prompt_path_for_low)
    prompt_dict_for_high = load_json_data(prompt_path_for_high)

    model = LLM(args)
    print("Preparing pipeline for inference...")
    model.prepare_for_inference()

    llm_rule_generate(
        args, paths, kg_rules_path, model, rdict, relation_regex,
        sampled_path_dict_only_relations, similar_rel_dict, sampled_path_with_valid_dir, rule_heads_without_path, prompt_info_dict, prompt_info_dict_ex1, prompt_info_zero_dict, prompt_dict_for_low, prompt_dict_for_high
    )


def llm_rule_generate(args, paths, kg_rules_path, model, rdict, relation_regex,
                      sampled_path_dict_only_relations, similiary_rel_dict, sampled_path_with_valid_dir, rule_heads_without_path, prompt_info_dict, prompt_info_dict_ex1, prompt_info_zero_dict, prompt_dict_for_low, prompt_dict_for_high):
    # Generate rules
    log_stage("生成规则：common / zero", 0, args.num_iter)
    common_query_and_rule = paths["common_query_and_rule"]
    zero_query_and_rule = paths["zero_query_and_rule"]

    # Optional: load the offline example pool once, and share it across worker threads.
    example_pool_index = None
    if getattr(args, "use_dynamic_examples", False):
        if args.example_pool_path is None:
            args.example_pool_path = maybe_windows_long_path(os.path.join(args.results_root_path, args.bat_file_name, "example_pool", args.dataset, "example_pool.jsonl"))
        example_pool_index = load_example_pool_index(args.example_pool_path)

    # Optional: load offline-built Relation Profiles once, and share across worker threads (common stage only).
    relation_profiles = None
    if getattr(args, "use_semantic_profile", False):
        profile_path = getattr(args, "semantic_profile_path", None)
        if profile_path is None:
            profile_path = maybe_windows_long_path(
                os.path.join(args.results_root_path, args.bat_file_name, "semantics", args.dataset, "relation_profile.json")
            )
            args.semantic_profile_path = profile_path
        loaded = load_relation_profiles(profile_path)
        relation_profiles = loaded if loaded else None

    if args.test_one:
        sampled_path_dict_only_relations_iterable = itertools.islice(
            sampled_path_dict_only_relations, args.test_common_num)
        sampled_path_dict_only_relations_total = min(
            args.test_common_num, len(sampled_path_dict_only_relations))
        rule_heads_without_path_iterable = itertools.islice(
            rule_heads_without_path, args.test_zero_num)
        rule_heads_without_path_total = min(
            args.test_zero_num, len(rule_heads_without_path))
    else:
        sampled_path_dict_only_relations_iterable = sampled_path_dict_only_relations
        sampled_path_dict_only_relations_total = len(sampled_path_dict_only_relations)
        rule_heads_without_path_iterable = rule_heads_without_path
        rule_heads_without_path_total = len(rule_heads_without_path)

    common_usage_stats = init_usage_summary()
    common_processed = 0
    sampled_rows = list(sampled_path_dict_only_relations_iterable)
    relation2id = rdict.rel2idx

    if args.is_zero:
        with ThreadPool(args.n) as p:
            with tqdm(
                total=len(sampled_rows),
                desc="LLM calls (common)",
                dynamic_ncols=True,
                file=sys.stdout,
            ) as llm_calls_progress:
                for usage in p.imap_unordered(
                    partial(
                        generate_rule,
                        rdict=rdict,
                        common_query_and_rule=common_query_and_rule,
                        kg_rules_path=kg_rules_path,
                        model=model,
                        args=args,
                        relation_regex=relation_regex,
                        similiary_rel_dict=similiary_rel_dict,
                        prompt_info_dict=prompt_info_dict,
                        prompt_info_dict_ex1=prompt_info_dict_ex1,
                        example_pool_index=example_pool_index,
                        progress_call=None,
                    ),
                    sampled_rows,
                ):
                    common_processed += 1
                    common_usage_stats = merge_usage_summary(common_usage_stats, usage)
                    llm_calls_progress.update(1)
                    update_progress_bar(
                        llm_calls_progress,
                        common_usage_stats,
                        common_processed,
                        sampled_path_dict_only_relations_total,
                    )
    else:
        common_head_states = {}
        common_tasks = []
        common_head_done = {}
        common_head_target = {}

        for row in sampled_rows:
            head_name = row["head"]
            all_paths_from_train = row["paths"]
            head_id = relation2id[head_name]
            head_formatted = head_name if args.is_rel_name is True else head_id

            readable_all_path_from_train = get_rule_format(head_name, all_paths_from_train, kg_rules_path)

            file_name = head_name.replace("/", "-")
            query_path = maybe_windows_long_path(os.path.join(common_query_and_rule, f"{file_name}.query"))
            txt_path = maybe_windows_long_path(os.path.join(common_query_and_rule, f"{file_name}.txt"))
            fail_path = maybe_windows_long_path(os.path.join(common_query_and_rule, f"fail_{file_name}.txt"))

            writer = _OrderedSampleWriter(
                query_path=query_path,
                txt_path=txt_path,
                fail_path=fail_path,
                include_rule_head=True,
                rule_head=head_name,
            )
            writer.init_files()

            if args.ex1_prompt is True:
                definition, role, examples_title, examples, sample_random_walk_paths_title, candidate_template, rule_head_title_for_user, generated_temporal_logic_rules_title = build_prompt_for_common_ex1(
                    head_formatted, prompt_info_dict_ex1
                )
                common_head_states[head_name] = {
                    "head_name": head_name,
                    "readable_all_path_from_train": readable_all_path_from_train,
                    "definition": definition,
                    "role": role,
                    "examples_title": examples_title,
                    "examples": examples,
                    "sample_random_walk_paths_title": sample_random_walk_paths_title,
                    "candidate_template": candidate_template,
                    "rule_head_title_for_user": rule_head_title_for_user,
                    "generated_temporal_logic_rules_title": generated_temporal_logic_rules_title,
                    "ex1_prompt": True,
                    "writer": writer,
                }
            else:
                definition, role, examples_title, examples, sampled_paths_title, goal, candidate_template, return_rules = build_prompt_for_common(
                    head_formatted, prompt_info_dict
                )
                common_head_states[head_name] = {
                    "head_name": head_name,
                    "readable_all_path_from_train": readable_all_path_from_train,
                    "definition": definition,
                    "role": role,
                    "examples_title": examples_title,
                    "examples": examples,
                    "sampled_paths_title": sampled_paths_title,
                    "goal": goal,
                    "candidate_template": candidate_template,
                    "return_rules": return_rules,
                    "ex1_prompt": False,
                    "writer": writer,
                }

            common_head_done[head_name] = 0
            common_head_target[head_name] = args.l
            for sample_idx in range(args.l):
                common_tasks.append((head_name, sample_idx))

        with ThreadPool(args.n) as p:
            with tqdm(
                total=len(common_tasks),
                desc="LLM calls (common)",
                dynamic_ncols=True,
                file=sys.stdout,
            ) as llm_calls_progress:
                for head_key, usage in p.imap_unordered(
                    partial(
                        _run_common_call_task,
                        head_states=common_head_states,
                        model=model,
                        args=args,
                        relation_regex=relation_regex,
                        similiary_rel_dict=similiary_rel_dict,
                        example_pool_index=example_pool_index,
                        profiles=relation_profiles,
                    ),
                    common_tasks,
                ):
                    common_usage_stats = merge_usage_summary(common_usage_stats, usage)
                    llm_calls_progress.update(1)
                    common_head_done[head_key] += 1
                    if common_head_done[head_key] >= common_head_target[head_key]:
                        common_processed += 1
                    update_progress_bar(
                        llm_calls_progress,
                        common_usage_stats,
                        common_processed,
                        sampled_path_dict_only_relations_total,
                    )

    zero_usage_stats = init_usage_summary()
    zero_processed = 0

    zero_heads = list(rule_heads_without_path_iterable)
    if zero_heads:
        all_rels_name = list(relation2id.keys())
        zero_head_states = {}
        zero_tasks = []
        zero_head_done = {}
        zero_head_target = {}

        for head in zero_heads:
            definition, role, examples_title, examples, goal, candidate_template, return_rules = build_prompt_for_zero(
                head, prompt_info_zero_dict
            )
            candidate = candidate_template.format(candidate_rels=all_rels_name)
            current_prompt = (
                definition + "\n\n" + role + "\n\n" + examples_title + "\n" + examples
                + "\n\n" + goal + "\n\n" + candidate + "\n\n" + return_rules
            )

            query_path = maybe_windows_long_path(os.path.join(zero_query_and_rule, f"{head}.query"))
            txt_path = maybe_windows_long_path(os.path.join(zero_query_and_rule, f"{head}.txt"))
            fail_path = maybe_windows_long_path(os.path.join(zero_query_and_rule, f"fail_{head}.txt"))

            writer = _OrderedSampleWriter(
                query_path=query_path,
                txt_path=txt_path,
                fail_path=fail_path,
                include_rule_head=False,
            )
            writer.init_files()

            zero_head_states[head] = {"prompt": current_prompt, "writer": writer}
            zero_head_done[head] = 0
            zero_head_target[head] = args.l
            for sample_idx in range(args.l):
                zero_tasks.append((head, sample_idx))

        with ThreadPool(args.n) as p:
            with tqdm(
                total=len(zero_tasks),
                desc="LLM calls (zero)",
                dynamic_ncols=True,
                file=sys.stdout,
            ) as llm_calls_progress:
                for head_key, usage in p.imap_unordered(
                    partial(
                        _run_zero_call_task,
                        head_states=zero_head_states,
                        model=model,
                        args=args,
                    ),
                    zero_tasks,
                ):
                    zero_usage_stats = merge_usage_summary(zero_usage_stats, usage)
                    llm_calls_progress.update(1)
                    zero_head_done[head_key] += 1
                    if zero_head_done[head_key] >= zero_head_target[head_key]:
                        zero_processed += 1
                    update_progress_bar(
                        llm_calls_progress,
                        zero_usage_stats,
                        zero_processed,
                        rule_heads_without_path_total,
                    )

    # valid dataset中的rule
    kg_rules_path_with_valid = maybe_windows_long_path(os.path.join(
        sampled_path_with_valid_dir, "rules_name.json"))


    constant_config = load_json_data(maybe_windows_long_path(os.path.join("Config", "constant.json")))
    relation_regex = constant_config['relation_regex'][args.dataset]

    # 初始化迭代第 0 轮的输入：合并 common/zero 的 txt 文件
    copy_folder_contents(common_query_and_rule, paths["iteration_list"][0])
    copy_folder_contents(zero_query_and_rule, paths["iteration_list"][0])

    output_filter_train_folder = paths["filter_train"]
    output_filter_eva_folder = paths["filter_eva"]
    output_filter_train_eva_folder = paths["filter_train_eva"]

    output_eva_train_folder = paths["evaluation_train"]
    output_eva_eva_folder = paths["evaluation_eva"]
    output_eva_train_eva_folder = paths["evaluation_train_eva"]

    for i in range(args.num_iter):

        log_stage("clean规则", i, args.num_iter)
        cleaned_rules_folder_dir = clean(args, model, paths["iteration_list"][i], i, paths)
        
        conf_folder = None
        if args.bgkg == 'train':
            log_stage("计算置信度并筛选：train", i, args.num_iter)
            train_rule_set = evaluation(args, cleaned_rules_folder_dir, output_eva_train_folder, 'train', index=i)
            if args.is_generate_pathes_for_train_dataset is True:
                return
            filter_rules_based_confidence(train_rule_set, args.min_conf, output_filter_train_folder, i)
            conf_folder = output_filter_train_folder
        elif args.bgkg == 'valid':
            log_stage("计算置信度并筛选：valid", i, args.num_iter)
            env_rule_set = evaluation(args, cleaned_rules_folder_dir, output_eva_eva_folder, 'eva', index=i)
            if args.is_generate_pathes_for_train_dataset is True:
                return
            filter_rules_based_confidence(env_rule_set, args.min_conf, output_filter_eva_folder, i)
            conf_folder = output_filter_eva_folder
        elif args.bgkg == 'train_valid':
            log_stage("计算置信度并筛选：train+valid", i, args.num_iter)
            train_env_rule_set = evaluation(args, cleaned_rules_folder_dir, output_eva_train_eva_folder, 'train_eva', index=i)
            if args.is_generate_pathes_for_train_dataset is True:
                return
            filter_rules_based_confidence(train_env_rule_set, args.min_conf, output_filter_train_eva_folder, i)
            conf_folder = output_filter_train_eva_folder

        if args.is_high is False:
            selected_conf_rules = get_low_conf(maybe_windows_long_path(os.path.join(
                conf_folder, f'{i}_temp_low_conf.txt')), relation_regex, rdict)
        else:
            selected_conf_rules = get_high_conf(maybe_windows_long_path(os.path.join(
                conf_folder, f'{i}_temp_hight_conf.txt')), relation_regex, rdict)
        log_stage("更新规则", i + 1, args.num_iter)
        gen_rules_iteration(args, kg_rules_path, model, rdict, relation_regex,
                            paths["iteration_list"][i + 1],
                            selected_conf_rules, similiary_rel_dict, kg_rules_path_with_valid, prompt_dict_for_low, prompt_dict_for_high)

    log_stage("clean规则", args.num_iter, args.num_iter)
    cleaned_rules_folder_dir = clean(
        args, model, paths["iteration_list"][args.num_iter], args.num_iter, paths)
    source_rule_path = None
    if args.bgkg == 'train':
        log_stage("计算置信度并筛选：train", args.num_iter, args.num_iter)
        train_rule_set = evaluation(
            args, cleaned_rules_folder_dir, output_eva_train_folder, 'train', index=args.num_iter)
        filter_rules_based_confidence(
            train_rule_set, args.min_conf, output_filter_train_folder, args.num_iter)
        analysis_data(output_filter_train_folder, kg_rules_path)
        source_rule_path = output_filter_train_folder
    elif args.bgkg == 'valid':
        log_stage("计算置信度并筛选：valid", args.num_iter, args.num_iter)
        env_rule_set = evaluation(
            args, cleaned_rules_folder_dir, output_eva_eva_folder, 'eva', index=args.num_iter)
        filter_rules_based_confidence(
            env_rule_set, args.min_conf, output_filter_eva_folder, args.num_iter)
        analysis_data(output_filter_eva_folder, kg_rules_path)
        source_rule_path = output_filter_eva_folder
    elif args.bgkg == 'train_valid':
        log_stage("计算置信度并筛选：train+valid", args.num_iter, args.num_iter)
        train_env_rule_set = evaluation(args, cleaned_rules_folder_dir, output_eva_train_eva_folder, 'train_eva',
                                        index=args.num_iter)
        filter_rules_based_confidence(
            train_env_rule_set, args.min_conf, output_filter_train_eva_folder, args.num_iter)
        analysis_data(output_filter_train_eva_folder, kg_rules_path)
        source_rule_path = output_filter_train_eva_folder

    if args.rule_domain == 'high':
        high_train_eva_file_path = maybe_windows_long_path(os.path.join(
            source_rule_path, 'hight_conf.txt'))
        with safe_open(high_train_eva_file_path, 'r', encoding="utf-8") as fin_high:
            high_unique_strings = set(fin_high.read().split())

        unique_strings = high_unique_strings

    elif args.rule_domain == 'iteration':
        high_train_eva_file_path = maybe_windows_long_path(os.path.join(
            source_rule_path, 'hight_conf.txt'))
        with safe_open(high_train_eva_file_path, 'r', encoding="utf-8") as fin_high:
            high_unique_strings = set(fin_high.read().split())

        low_train_eva_file_path = maybe_windows_long_path(os.path.join(
            source_rule_path, 'low_conf.txt'))
        with safe_open(low_train_eva_file_path, 'r', encoding="utf-8") as fin_low:
            low_unique_strings = set(fin_low.read().split())
        log_stage("合并高低置信度规则")
        unique_strings = low_unique_strings.union(high_unique_strings)

    else:
        pass
    
    log_stage("写入final_summary文件")
    with safe_open(maybe_windows_long_path(os.path.join(paths["final_summary"], "rules.txt")), "w", encoding="utf-8") as fout_final:
        for rule in sorted(unique_strings):
            fout_final.write(f"{rule}\n")


def gen_rules_iteration(args, kg_rules_path, model, rdict, relation_regex, rule_path, selected_conf_rules,
                        similiary_rel_dict, kg_rules_path_with_valid, prompt_dict_for_low, prompt_dict_for_high):
    if args.test_one:
        selected_conf_rules_list = list(itertools.islice(selected_conf_rules, args.test_iter_num))
    else:
        selected_conf_rules_list = list(selected_conf_rules)
    selected_conf_rules_total = len(selected_conf_rules_list)

    if selected_conf_rules_total <= 0:
        return

    # Keep the legacy guard (also present in the old per-head implementation).
    if args.k == 0 and args.is_zero:
        raise NotImplementedError(
            "Cannot implement for zero-shot(f=0) and generate zero(k=0) rules."
        )

    iteration_usage_stats = init_usage_summary()
    iteration_processed = 0

    relation2id = rdict.rel2idx
    valid_paths_dict = load_json_data(kg_rules_path_with_valid) or {}
    sampled_paths_dict_from_train = load_json_data(kg_rules_path) or {}

    if args.is_rel_name is True:
        all_rels = list(relation2id.keys())
        candidate_rels_all = ", ".join(all_rels)
    else:
        all_rels = list(relation2id.values())
        candidate_rels_all = ", ".join([str(item) for item in all_rels])

    based_rule_type = getattr(args, "based_rule_type", "low")
    based_rule_type = "high" if based_rule_type != "low" else "low"

    iteration_head_states = {}
    iteration_tasks = []
    iteration_head_done = {}
    iteration_head_target = {}

    for selected_conf_rule_for_one_head in selected_conf_rules_list:
        head_name = selected_conf_rule_for_one_head["head"]
        selected_rules = selected_conf_rule_for_one_head.get("rules", []) or []
        head_id = relation2id[head_name]
        head_formate = head_name if args.is_rel_name is True else head_id

        file_name = head_name.replace("/", "-")
        query_path = maybe_windows_long_path(os.path.join(rule_path, f"{file_name}.query"))
        txt_path = maybe_windows_long_path(os.path.join(rule_path, f"{file_name}.txt"))
        fail_path = maybe_windows_long_path(os.path.join(rule_path, f"fail_{file_name}.txt"))

        writer = _OrderedSampleWriter(
            query_path=query_path,
            txt_path=txt_path,
            fail_path=fail_path,
            include_rule_head=True,
            rule_head=head_name,
        )
        writer.init_files()

        if based_rule_type == "low":
            definition, role, examples_title, examples, low_quality_rules_title, sample_rules_title, goal, candidate_template, return_rules = build_prompt_based_low(
                head_formate, candidate_rels_all, args.is_zero, args, prompt_dict_for_low
            )
            iteration_head_states[head_name] = {
                "based_rule_type": "low",
                "selected_head_name": head_name,
                "selected_conf_rules": selected_rules,
                "all_rels": all_rels,
                "head_formate": head_formate,
                "candidate_rels_all": candidate_rels_all,
                "definition": definition,
                "role": role,
                "examples_title": examples_title,
                "examples": examples,
                "low_quality_rules_title": low_quality_rules_title,
                "sample_rules_title": sample_rules_title,
                "goal": goal,
                "candidate_template": candidate_template,
                "return_rules": return_rules,
                "writer": writer,
            }
        else:
            iteration_head_states[head_name] = {
                "based_rule_type": "high",
                "selected_head_name": head_name,
                "head_formate": head_formate,
                "candidate_rels_all": candidate_rels_all,
                "writer": writer,
            }

        iteration_head_done[head_name] = 0
        iteration_head_target[head_name] = args.second
        for sample_idx in range(args.second):
            iteration_tasks.append((head_name, sample_idx))

    with ThreadPool(args.n) as p:
        with tqdm(
            total=len(iteration_tasks),
            desc="LLM calls (iteration)",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as llm_calls_progress:
            for head_key, usage in p.imap_unordered(
                partial(
                    _run_iteration_call_task,
                    head_states=iteration_head_states,
                    model=model,
                    args=args,
                    relation_regex=relation_regex,
                    similiary_rel_dict=similiary_rel_dict,
                    sampled_paths_dict_from_train=sampled_paths_dict_from_train,
                    valid_paths_dict=valid_paths_dict,
                    prompt_dict_for_low=prompt_dict_for_low,
                    prompt_dict_for_high=prompt_dict_for_high,
                ),
                iteration_tasks,
            ):
                iteration_usage_stats = merge_usage_summary(iteration_usage_stats, usage)
                llm_calls_progress.update(1)
                iteration_head_done[head_key] += 1
                if iteration_head_done[head_key] >= iteration_head_target[head_key]:
                    iteration_processed += 1
                update_progress_bar(
                    llm_calls_progress,
                    iteration_usage_stats,
                    iteration_processed,
                    selected_conf_rules_total,
                )



def filter_rules_based_confidence(rule_set, min_conf, output_folder, index):
    with safe_open(os.path.join(output_folder, 'hight_conf.txt'), 'a', encoding="utf-8") as fout_hight, safe_open(
        os.path.join(output_folder, 'low_conf.txt'), 'a', encoding="utf-8") as fout_low, safe_open(
            os.path.join(output_folder, f'{index}_temp_hight_conf.txt'), 'w', encoding="utf-8") as fout_temp_hight, safe_open(
            os.path.join(output_folder, f'{index}_temp_low_conf.txt'), 'w', encoding="utf-8") as fout_temp_low:
        fout_hight.write(f"index:{index}\n")
        fout_low.write(f"index:{index}\n")
        for rule in sorted(rule_set):
            confidence = float(rule.split('&')[-1].strip())
            temp_rule = rule.split('&')[:-1]
            rule_without_confidence = '&'.join(temp_rule)
            if confidence > min_conf:
                fout_hight.write(rule_without_confidence + '\n')
                fout_temp_hight.write(rule_without_confidence + '\n')
            else:
                fout_low.write(rule_without_confidence + '\n')
                fout_temp_low.write(rule_without_confidence + '\n')


def evaluation(args, output_rules_folder_dir, output_evaluation_folder, dataset_type, index=0):
    is_merge = args.is_merge
    dataset_dir = maybe_windows_long_path(os.path.join("datasets", args.dataset))
    data = Grapher(dataset_dir)

    if dataset_type == 'train':
        temporal_walk = Temporal_Walk(np.array(data.train_idx.tolist()), data.inv_relation_id,
                                      args.transition_distr)
    elif dataset_type == 'eva':
        temporal_walk = Temporal_Walk(np.array(data.valid_idx.tolist()), data.inv_relation_id,
                                      args.transition_distr)
    else:
        temporal_walk = Temporal_Walk(np.array(data.valid_idx.tolist() + data.train_idx.tolist()), data.inv_relation_id,
                                      args.transition_distr)

    rl = Rule_Learner(temporal_walk.edges, data.id2relation,
                      data.inv_relation_id, args.dataset)
    rule_path = output_rules_folder_dir
    constant_config = load_json_data(maybe_windows_long_path(os.path.join("Config", "constant.json")))
    relation_regex = constant_config['relation_regex'][args.dataset]

    rules_var_path = maybe_windows_long_path(os.path.join(
        "sampled_path", args.dataset, "original", "rules_var.json"))
    rules_var_dict = load_json_data(rules_var_path)

    if args.is_only_with_original_rules:
        for key, value in rules_var_dict.items():
            temp_var = {}
            temp_var['head_rel'] = value['head_rel']
            temp_var['body_rels'] = value['body_rels']
            temp_var["var_constraints"] = value["var_constraints"]
            if temp_var not in rl.original_found_rules:
                rl.original_found_rules.append(temp_var.copy())
                rl.update_rules_dict(value)
                rl.num_original += 1
    else:
        llm_gen_rules_list, fail_calc_confidence = calculate_confidence(
            rule_path,
            data.relation2id,
            data.inv_relation_id,
            rl,
            relation_regex,
            rules_var_dict,
            is_merge,
            is_has_confidence=False,
            eval_workers=getattr(args, "eval_workers", 1),
            chunksize=getattr(args, "eval_chunksize", 8),
            base_seed=getattr(args, "base_seed", 0),
            dataset_name=args.dataset,
        )

    rules_statistics(rl.rules_dict)
    rules_dict_for_save = stable_sort_rules_dict(rl.rules_dict)

    if args.is_only_with_original_rules:
        dir_path = output_evaluation_folder
        confidence_file_path = maybe_windows_long_path(os.path.join(
            dir_path, 'original_confidence.json'))
        save_json_data(rules_dict_for_save, confidence_file_path)
    else:
        if is_merge is True:
            original_rules_set = set(list(rules_var_dict.keys()))
            llm_gen_rules_set = set(llm_gen_rules_list)
            for rule_chain in sorted(original_rules_set - llm_gen_rules_set):
                rule = rules_var_dict[rule_chain]
                rl.update_rules_dict(rule)

            rules_statistics(rl.rules_dict)
            rules_dict_for_save = stable_sort_rules_dict(rl.rules_dict)

            dir_path = output_evaluation_folder
            confidence_file_path = maybe_windows_long_path(os.path.join(
                dir_path, 'merge_confidence.json'))
            save_json_data(rules_dict_for_save, confidence_file_path)
        else:
            dir_path = output_evaluation_folder
            confidence_file_path = maybe_windows_long_path(os.path.join(
                dir_path, f'{index}_confidence.json'))
            save_json_data(rules_dict_for_save, confidence_file_path)
            confidence_concrete_path = maybe_windows_long_path(os.path.join(
                dir_path, f'{index}_confidence_concrete.json'))
            confidence_with_names = build_confidence_with_names(
                rules_dict_for_save, data.id2relation)
            save_json_data(confidence_with_names, confidence_concrete_path)

            fail_confidence_file_path = maybe_windows_long_path(os.path.join(
                dir_path, f'{index}_fail_confidence.txt'))
            with safe_open(fail_confidence_file_path, 'a', encoding="utf-8") as fout:
                for fail_rule in sorted(fail_calc_confidence):
                    fout.write(f'{fail_rule}\n')

    return set(llm_gen_rules_list)


def derive_seed_64(base_seed: int, *parts: object) -> int:
    """
    Deterministically derive a 64-bit seed from (base_seed, parts...).
    Stable across runs & machines (does not depend on PYTHONHASHSEED).
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(base_seed)).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False)


def derive_seed_32(base_seed: int, *parts: object) -> int:
    """
    Deterministically derive a 32-bit seed from (base_seed, parts...).
    Stable across runs & machines.
    """
    return derive_seed_64(base_seed, *parts) % (2 ** 32)


def get_task_rng(base_seed: int, *parts: object):
    """
    Return a per-task RNG object with `.sample()` / `.shuffle()`:
      - base_seed == 0 => global `random` module (non-reproducible mode)
      - base_seed != 0 => `random.Random(derive_seed_64(...))` (reproducible)
    """
    if not base_seed:
        return random
    return random.Random(derive_seed_64(base_seed, *parts))


def format_candidate_relations(candidate_relations, *, all_rels_order=None) -> str:
    """
    Stable stringify helper for candidate relations.
    - If all_rels_order is given, keep that order.
    - Else, sort lexicographically.
    """
    s = set(candidate_relations or [])
    if all_rels_order is not None:
        ordered = [r for r in all_rels_order if r in s]
    else:
        ordered = sorted(s)
    return ";".join(str(x) for x in ordered)


def _confidence_rule_key(rule_dict):
    head = int(rule_dict.get("head_rel"))
    body = tuple(int(x) for x in (rule_dict.get("body_rels") or []))
    vc = rule_dict.get("var_constraints") or []
    vc_norm = []
    for grp in vc:
        if isinstance(grp, (list, tuple)):
            vc_norm.append(tuple(int(i) for i in grp))
        else:
            vc_norm.append((int(grp),))
    return (head, body, tuple(vc_norm))


def _compute_confidence_for_file(
    input_filepath,
    relation2id,
    inv_relation_id,
    relation_regex,
    rl,
    rules_var_dict=None,
    is_merge=False,
    is_has_confidence=False,
    base_seed=0,
):
    start = time.time()
    llm_gen_rules_list = []
    fail_calc_confidence = []
    rule_dicts = []

    with safe_open(input_filepath, "r", encoding="utf-8") as f:
        rules = f.readlines()

    seen = set()
    for rule in rules:
        raw = (rule or "").strip()
        if not raw:
            continue

        try:
            if is_has_confidence:
                confidence = float(raw.split("&")[-1].strip())
                rule_without_confidence = "&".join(raw.split("&")[:-1]).strip()
            else:
                confidence = 0.0
                rule_without_confidence = raw

            if not rule_without_confidence:
                continue
            if rule_without_confidence in seen:
                continue
            seen.add(rule_without_confidence)

            if is_merge and rules_var_dict is not None and rules_var_dict.get(rule_without_confidence) is not None:
                rule_obj = copy.deepcopy(rules_var_dict[rule_without_confidence])
                rule_obj["llm_confidence"] = confidence
                conf_value = rule_obj.get("conf", 0)
                llm_gen_rules_list.append(rule_without_confidence + "&" + str(conf_value) + "\n")
                if conf_value or confidence:
                    rule_dicts.append(rule_obj)
                continue

            if base_seed:
                np.random.seed(derive_seed_32(base_seed, rule_without_confidence))

            walk = get_walk(rule_without_confidence, relation2id, inv_relation_id, relation_regex)

            rule_obj = dict()
            rule_obj["head_rel"] = int(walk["relations"][0])
            rule_obj["body_rels"] = [inv_relation_id[x] for x in walk["relations"][1:][::-1]]
            rule_obj["var_constraints"] = rl.define_var_constraints(walk["entities"][1:][::-1])
            rule_obj["conf"], rule_obj["rule_supp"], rule_obj["body_supp"] = rl.estimate_confidence(rule_obj)

            llm_gen_rules_list.append(rule_without_confidence + "&" + str(rule_obj["conf"]) + "\n")

            if rule_obj["body_supp"] == 0:
                rule_obj["body_supp"] = 2
            rule_obj["llm_confidence"] = confidence

            if rule_obj["conf"] or confidence:
                rule_dicts.append(rule_obj)
        except Exception:
            fail_calc_confidence.append(rule + "\n")

    elapsed = time.time() - start
    return llm_gen_rules_list, fail_calc_confidence, rule_dicts, elapsed


_CONF_WORKER_RL = None
_CONF_WORKER_RELATION2ID = None
_CONF_WORKER_INV_RELATION_ID = None
_CONF_WORKER_RELATION_REGEX = None
_CONF_WORKER_RULES_VAR_DICT = None
_CONF_WORKER_IS_MERGE = False
_CONF_WORKER_BASE_SEED = 0


def _init_confidence_worker(edges, id2relation, inv_relation_id, dataset_name,
                            relation2id, relation_regex, rules_var_dict,
                            is_merge, base_seed):
    global _CONF_WORKER_RL, _CONF_WORKER_RELATION2ID, _CONF_WORKER_INV_RELATION_ID, _CONF_WORKER_RELATION_REGEX
    global _CONF_WORKER_RULES_VAR_DICT, _CONF_WORKER_IS_MERGE, _CONF_WORKER_BASE_SEED

    _CONF_WORKER_RELATION2ID = relation2id
    _CONF_WORKER_INV_RELATION_ID = inv_relation_id
    _CONF_WORKER_RELATION_REGEX = relation_regex
    _CONF_WORKER_RULES_VAR_DICT = rules_var_dict
    _CONF_WORKER_IS_MERGE = bool(is_merge)
    _CONF_WORKER_BASE_SEED = int(base_seed or 0)

    _CONF_WORKER_RL = Rule_Learner(edges, id2relation, inv_relation_id, dataset_name)

    if _CONF_WORKER_BASE_SEED == 0:
        np.random.seed(int.from_bytes(os.urandom(4), "little", signed=False))
    else:
        np.random.seed(_CONF_WORKER_BASE_SEED % (2 ** 32))


def _confidence_worker_process_file(task):
    input_filepath, is_has_confidence = task
    file_llm, file_fail, rule_dicts, elapsed = _compute_confidence_for_file(
        input_filepath,
        _CONF_WORKER_RELATION2ID,
        _CONF_WORKER_INV_RELATION_ID,
        _CONF_WORKER_RELATION_REGEX,
        _CONF_WORKER_RL,
        rules_var_dict=_CONF_WORKER_RULES_VAR_DICT,
        is_merge=_CONF_WORKER_IS_MERGE,
        is_has_confidence=is_has_confidence,
        base_seed=_CONF_WORKER_BASE_SEED,
    )
    return input_filepath, file_llm, file_fail, rule_dicts, elapsed


def calculate_confidence(rule_path, relation2id, inv_relation_id, rl, relation_regex, rules_var_dict, is_merge,
                         is_has_confidence=False, eval_workers=1, chunksize=8, base_seed=0, dataset_name=None):
    llm_gen_rules_list = []
    fail_calc_confidence = []
    file_list = sorted(
        glob.glob(maybe_windows_long_path(os.path.join(rule_path, "*_cleaned_rules.txt"))),
        key=str.lower,
    )

    try:
        eval_workers = int(eval_workers)
    except Exception:
        eval_workers = 1
    if eval_workers <= 0:
        eval_workers = max(1, (os.cpu_count() or 1))

    try:
        chunksize = int(chunksize)
    except Exception:
        chunksize = 8
    if chunksize <= 0:
        chunksize = 1

    try:
        base_seed = int(base_seed)
    except Exception:
        base_seed = 0

    if not dataset_name:
        dataset_name = "unknown"

    seen_rule_keys = set()
    elapsed_sum = 0.0

    use_parallel = (eval_workers > 1) and (len(file_list) > 1)
    if not use_parallel:
        with tqdm(total=len(file_list), desc="计算置信度", dynamic_ncols=True, file=sys.stdout) as progress:
            for input_filepath in file_list:
                file_llm, file_fail, rule_dicts, elapsed = _compute_confidence_for_file(
                    input_filepath,
                    relation2id,
                    inv_relation_id,
                    relation_regex,
                    rl,
                    rules_var_dict=rules_var_dict if is_merge else None,
                    is_merge=is_merge,
                    is_has_confidence=is_has_confidence,
                    base_seed=base_seed,
                )
                llm_gen_rules_list.extend(file_llm)
                fail_calc_confidence.extend(file_fail)

                for rule_obj in rule_dicts:
                    k = _confidence_rule_key(rule_obj)
                    if k in seen_rule_keys:
                        continue
                    seen_rule_keys.add(k)
                    rl.update_rules_dict(rule_obj)

                elapsed_sum += elapsed
                processed = progress.n + 1
                avg_t = elapsed_sum / processed if processed else 0
                progress.set_postfix({"avg_t": format_seconds(avg_t)}, refresh=False)
                progress.update(1)
        return llm_gen_rules_list, fail_calc_confidence

    ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context()
    initargs = (
        rl.edges,
        rl.id2relation,
        inv_relation_id,
        dataset_name,
        relation2id,
        relation_regex,
        rules_var_dict if is_merge else None,
        is_merge,
        base_seed,
    )
    tasks = [(fp, is_has_confidence) for fp in file_list]

    with tqdm(total=len(file_list), desc="计算置信度", dynamic_ncols=True, file=sys.stdout) as progress:
        results_by_file = {}
        with ctx.Pool(processes=eval_workers, initializer=_init_confidence_worker, initargs=initargs) as pool:
            for input_filepath, file_llm, file_fail, rule_dicts, elapsed in pool.imap_unordered(
                _confidence_worker_process_file, tasks, chunksize=chunksize
            ):
                results_by_file[input_filepath] = (file_llm, file_fail, rule_dicts, elapsed)

                elapsed_sum += elapsed
                processed = progress.n + 1
                avg_t = elapsed_sum / processed if processed else 0
                progress.set_postfix({"avg_t": format_seconds(avg_t)}, refresh=False)
                progress.update(1)

    # Deterministic merge (sorted file_list order) to make dedup stable.
    for input_filepath in file_list:
        file_llm, file_fail, rule_dicts, _elapsed = results_by_file.get(input_filepath, ([], [], [], 0.0))
        llm_gen_rules_list.extend(file_llm)
        fail_calc_confidence.extend(file_fail)

        for rule_obj in rule_dicts:
            k = _confidence_rule_key(rule_obj)
            if k in seen_rule_keys:
                continue
            seen_rule_keys.add(k)
            rl.update_rules_dict(rule_obj)

    return llm_gen_rules_list, fail_calc_confidence




def get_walk(rule, relation2id, inv_relation_id, regex):
    head_body = rule.split('<-')
    rule_head_full_name = head_body[0].strip()
    condition_string = head_body[1].strip()

    # 定义正则表达式
    relation_regex = regex

    # 提取规则头的关系、主语和宾语
    match = re.search(relation_regex, rule_head_full_name)
    head_relation_name, head_subject, head_object, head_timestamp = match.groups()[
        :4]

    # 提取规则体的关系和实体
    matches = re.findall(relation_regex, condition_string)
    entities = [head_object] + [match[1].strip() for match in matches[:-1]] + [matches[-1][1].strip(),
                                                                               matches[-1][2].strip()]

    relation_ids = [relation2id[head_relation_name]] + \
        [relation2id[match[0].strip()] for match in matches]

    # 反转除第一个元素外的列表
    entities = entities[:1] + entities[1:][::-1]
    relation_ids = relation_ids[:1] + [inv_relation_id[x]
                                       for x in relation_ids[:0:-1]]

    # 构造结果字典
    result = {
        'entities': entities,
        'relations': relation_ids
    }

    return result


def clean(args, llm_model, rules_folder_to_clean, index, paths):
    data_path = maybe_windows_long_path(os.path.join(args.data_path, args.dataset))
    dataset = Dataset(data_root=data_path, inv=True)
    rdict = dataset.get_relation_dict()
    all_rels = list(rdict.rel2idx.keys())

    cleaned_statistics_folder_dir = paths["cleaned_statistics_list"][index]
    cleaned_rules_folder_dir = paths["cleaned_rules_list"][index]

    # 分析clean过程中success与error的情况
    cleaned_statistics_error_file_path = maybe_windows_long_path(os.path.join(cleaned_statistics_folder_dir, 'error.txt'))
    cleaned_statistics_suc_file_path = maybe_windows_long_path(os.path.join(cleaned_statistics_folder_dir, 'suc.txt'))
    with safe_open(cleaned_statistics_error_file_path, 'w', encoding="utf-8") as fout_error, safe_open(cleaned_statistics_suc_file_path, 'w', encoding="utf-8") as fout_suc:
        num_error, num_suc = clean_processing(all_rels, args, fout_error, rules_folder_to_clean, llm_model,
                                              cleaned_rules_folder_dir,
                                              fout_suc)
        fout_error.write(f"The number of cleaned rules is {num_error}\n")
        fout_suc.write(f"The number of retain rules is {num_suc}\n")

    return cleaned_rules_folder_dir


def clean_processing(all_rels, args, fout_error, rules_folder_to_clean, llm_model, cleaned_rules_folder_dir, fout_suc):
    constant_config = load_json_data(maybe_windows_long_path(os.path.join("Config", "constant.json")))
    rule_start_with_regex = constant_config["rule_start_with_regex"]
    replace_regex = constant_config["replace_regex"]
    relation_regex = constant_config["relation_regex"][args.dataset]
    num_error = 0
    num_suc = 0
    file_list = sorted(
        [
            f for f in os.listdir(rules_folder_to_clean)
            if f.endswith(".txt") and "query" not in f and not f.startswith("fail")
        ],
        key=str.lower,
    )
    clean_workers = getattr(args, "clean_workers", 1)
    try:
        clean_workers = int(clean_workers)
    except Exception:
        clean_workers = 1
    if clean_workers <= 0:
        clean_workers = max(1, min(32, (os.cpu_count() or 1)))
    elapsed_sum = 0.0
    if clean_workers <= 1 or len(file_list) <= 1:
        with tqdm(total=len(file_list), desc="Clean & summarize", dynamic_ncols=True, file=sys.stdout) as progress:
            for filename in file_list:
                start = time.time()
                input_filepath = maybe_windows_long_path(os.path.join(rules_folder_to_clean, filename))
                name, ext = os.path.splitext(filename)
                summarized_filepath = maybe_windows_long_path(os.path.join(
                    cleaned_rules_folder_dir, f"{name}_summarized_rules.txt"))
                clean_filename = name + '_cleaned_rules.txt'
                clean_filepath = maybe_windows_long_path(os.path.join(cleaned_rules_folder_dir, clean_filename))

                if not args.clean_only:
                    summarized_rules = summarize_rule(
                        input_filepath, llm_model, args, rule_start_with_regex, replace_regex)
                    with safe_open(summarized_filepath, "w", encoding="utf-8") as f:
                        f.write('\n'.join(summarized_rules))

                cleaned_rules, num, num_0 = clean_rules(
                    summarized_filepath, all_rels, relation_regex, fout_error, fout_suc)
                num_error = num_error + num
                num_suc = num_suc + num_0

                if len(cleaned_rules) != 0:
                    with safe_open(clean_filepath, "w", encoding="utf-8") as f:
                        f.write('\n'.join(cleaned_rules))

                elapsed_sum += time.time() - start
                processed = progress.n + 1
                avg_t = elapsed_sum / processed if processed else 0
                progress.set_postfix(
                    {"avg_t": format_seconds(avg_t)},
                    refresh=False,
                )
                progress.update(1)
    else:
        stats_dir = None
        try:
            stats_dir = os.path.dirname(getattr(fout_error, "name", "") or "")
        except Exception:
            stats_dir = None
        if not stats_dir:
            try:
                stats_dir = os.path.dirname(getattr(fout_suc, "name", "") or "")
            except Exception:
                stats_dir = None
        if not stats_dir:
            stats_dir = cleaned_rules_folder_dir
        stats_dir = maybe_windows_long_path(stats_dir)

        def _clean_one_file(filename):
            start = time.time()
            input_filepath = maybe_windows_long_path(os.path.join(rules_folder_to_clean, filename))
            name, ext = os.path.splitext(filename)
            summarized_filepath = maybe_windows_long_path(os.path.join(
                cleaned_rules_folder_dir, f"{name}_summarized_rules.txt"))
            clean_filename = name + '_cleaned_rules.txt'
            clean_filepath = maybe_windows_long_path(os.path.join(cleaned_rules_folder_dir, clean_filename))

            if not args.clean_only:
                summarized_rules = summarize_rule(
                    input_filepath, llm_model, args, rule_start_with_regex, replace_regex)
                with safe_open(summarized_filepath, "w", encoding="utf-8") as f:
                    f.write('\n'.join(summarized_rules))

            error_part_path = maybe_windows_long_path(os.path.join(stats_dir, f"{name}_error_part.txt"))
            suc_part_path = maybe_windows_long_path(os.path.join(stats_dir, f"{name}_suc_part.txt"))
            with safe_open(error_part_path, "w", encoding="utf-8") as ferr, safe_open(suc_part_path, "w", encoding="utf-8") as fsuc:
                cleaned_rules, num, num_0 = clean_rules(
                    summarized_filepath, all_rels, relation_regex, ferr, fsuc)

            if len(cleaned_rules) != 0:
                with safe_open(clean_filepath, "w", encoding="utf-8") as f:
                    f.write('\n'.join(cleaned_rules))

            return {
                "elapsed": time.time() - start,
                "num_error": num,
                "num_suc": num_0,
                "error_part_path": error_part_path,
                "suc_part_path": suc_part_path,
            }

        with tqdm(total=len(file_list), desc="Clean & summarize", dynamic_ncols=True, file=sys.stdout) as progress:
            with ThreadPoolExecutor(max_workers=clean_workers) as executor:
                future_to_name = {executor.submit(_clean_one_file, filename): filename for filename in file_list}
                results_by_name = {}
                for fut in as_completed(list(future_to_name.keys())):
                    filename = future_to_name[fut]
                    result = fut.result()
                    results_by_name[filename] = result
                    num_error += result["num_error"]
                    num_suc += result["num_suc"]

                    elapsed_sum += result["elapsed"]
                    processed = progress.n + 1
                    avg_t = elapsed_sum / processed if processed else 0
                    progress.set_postfix(
                        {"avg_t": format_seconds(avg_t)},
                        refresh=False,
                    )
                    progress.update(1)

        # Deterministic merge (by sorted file_list order)
        for filename in file_list:
            r = results_by_name.get(filename)
            if not r:
                continue
            err_path = r.get("error_part_path")
            suc_path = r.get("suc_part_path")
            if err_path and os.path.exists(err_path):
                with safe_open(err_path, "r", encoding="utf-8") as fin:
                    shutil.copyfileobj(fin, fout_error)
                try:
                    os.remove(err_path)
                except Exception:
                    pass
            if suc_path and os.path.exists(suc_path):
                with safe_open(suc_path, "r", encoding="utf-8") as fin:
                    shutil.copyfileobj(fin, fout_suc)
                try:
                    os.remove(suc_path)
                except Exception:
                    pass
    return num_error, num_suc


def extract_rules(content_list, rule_start_with_regex, replace_regex):
    """ Extract the rules in the content without any explanation and the leading number if it has."""
    rule_pattern = re.compile(rule_start_with_regex)
    extracted_rules = [s.strip()
                       for s in content_list if rule_pattern.match(s)]
    number_pattern = re.compile(replace_regex)
    cleaned_rules = [number_pattern.sub('', s) for s in extracted_rules]
    # Stable dedup (preserve first occurrence order)
    return list(dict.fromkeys(cleaned_rules))


def summarize_rules_prompt(relname, k):
    """
    Generate prompt for the relation in the content_list
    """

    if k != 0:
        prompt = f'\n\nPlease identify the most important {k} rules from the following rules for the rule head: "{relname}(X,Y,T)". '
    else:  # k ==0
        prompt = f'\n\nPlease identify as many of the most important rules for the rule head: "{relname}(X,Y,T)" as possible. '

    prompt += 'You can summarize the rules that have similar meanings as one rule, if you think they are important. ' \
              'Return the rules only without any explanations. '
    return prompt



def summarize_rule(file, llm_model, args, rule_start_with_regex, replace_regex):
    """
    Summarize the rules
    """
    with safe_open(file, 'r', encoding="utf-8") as f:  # Load files
        content = f.read()
        rel_name = os.path.splitext(os.path.basename(file))[0]

    content_list = content.split('\n')
    rule_list = extract_rules(content_list, rule_start_with_regex,
                              replace_regex)  # Extract rules and remove any explanations
    if not args.force_summarize or llm_model is None:  # just return the whole rule_list
        return rule_list
    else:  # Do summarization and correct the spelling error
        summarize_prompt = summarize_rules_prompt(rel_name, args.k)
        summarize_prompt_len = num_tokens_from_message(
            summarize_prompt, args.model_name)
        rng_shuffle = get_task_rng(getattr(args, "base_seed", 0), "clean", rel_name, "summarize_shuffle")
        list_of_rule_lists = shuffle_split_path_list(
            rule_list, summarize_prompt_len, args.model_name, rng=rng_shuffle)
        response_list = []
        for rule_list in list_of_rule_lists:
            message = '\n'.join(rule_list) + summarize_prompt
            response = query(message, llm_model)
            response_list.extend(response.split('\n'))
        response_rules = extract_rules(response_list, rule_start_with_regex,
                                       replace_regex)  # Extract rules and remove any explanations from summarized response

        return response_rules



def clean_rules(summarized_file_path, all_rels, relation_regex, fout_error, fout_suc):
    """
    Clean error rules and remove rules with error relation.
    """
    num_error = 0
    num_suc = 0
    with safe_open(summarized_file_path, 'r', encoding="utf-8") as f:
        input_rules = [line.strip() for line in f]
    cleaned_rules = list()
    # Correct spelling error/grammar error for the relation in the rules and Remove rules with error relation.
    for input_rule in input_rules:
        if input_rule == "":
            continue
        rule_list = []
        temp_rule = re.sub(r'\s*<-\s*', '&', input_rule)
        regrex_list = temp_rule.split('&')
        last_subject = None
        final_object = None
        time_squeque = []
        final_time = None
        is_save = True
        try:
            for idx, regrex in enumerate(regrex_list):
                match = re.search(relation_regex, regrex)
                if match:
                    relation_name = match[1].strip()
                    subject = match[2].strip()
                    object = match[3].strip()
                    timestamp = match[4].strip()

                    if timestamp[1:].isdigit() is False:
                        fout_error.write(
                            f"Error: Rule {input_rule}:{timestamp} is not digit\n")
                        num_error = num_error + 1
                        is_save = False
                        break

                    if relation_name not in all_rels:
                        best_match = get_close_matches(
                            relation_name, all_rels, n=1)
                        if not best_match:
                            fout_error.write(
                                f"Cannot correctify this rule, head not in relation:{input_rule}\n")
                            is_save = False
                            num_error = num_error + 1
                            break
                        relation_name = best_match[0].strip()

                    rule_list.append(
                        f'{relation_name}({subject},{object},{timestamp})')

                    if idx == 0:
                        head_subject = subject
                        head_object = object
                        head_subject = head_subject

                        last_subject = head_subject
                        final_object = head_object

                        final_time = int(timestamp[1:])
                    else:
                        if last_subject == subject:
                            last_subject = object
                        else:
                            fout_error.write(
                                f"Error: Rule {input_rule} does not conform to the definition of chain rule.\n")
                            num_error = num_error + 1
                            is_save = False
                            break

                        time_squeque.append(int(timestamp[1:]))

                    if idx == len(regrex_list) - 1:
                        if last_subject != final_object:
                            fout_error.write(
                                f"Error: Rule {input_rule} does not conform to the definition of chain rule.\n")
                            num_error = num_error + 1
                            is_save = False
                            break

                else:
                    fout_error.write(f"Error: rule {input_rule}\n")
                    num_error = num_error + 1
                    is_save = False
                    break

            if all(time_squeque[i] <= time_squeque[i + 1] for i in range(len(time_squeque) - 1)) is False:
                fout_error.write(
                    f"Error: Rule {input_rule} time_squeque is error.\n")
                num_error = num_error + 1
                is_save = False
            elif final_time < time_squeque[-1]:
                fout_error.write(
                    f"Error: Rule {input_rule} time_squeque is error.\n")
                num_error = num_error + 1
                is_save = False

            if is_save:
                correct_rule = '&'.join(
                    rule_list).strip().replace('&', '<-', 1)
                cleaned_rules.append(correct_rule)
                fout_suc.write(correct_rule + '\n')
                num_suc = num_suc + 1

        except Exception as e:
            fout_error.write(
                f"Processing {input_rule} failed.\n Error: {str(e)}\n")
            num_error = num_error + 1
    return cleaned_rules, num_error, num_suc


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="KGC rule generation parameters")
    parser.add_argument("--data_path", type=str,
                        default="datasets", help="Data directory")
    parser.add_argument("--dataset", type=str,
                        default="family", help="Dataset name")
    parser.add_argument("--sampled_paths", type=str,
                        default="sampled_path", help="Sampled path directory")
    parser.add_argument("--prompt_paths", type=str,
                        default="prompt", help="Sampled path directory")
    parser.add_argument(
        "--use_dynamic_examples",
        action="store_true",
        help="Enable dynamic examples: retrieve few-shot examples from a local JSONL example pool (replaces prompt/common.json fixed examples).",
    )
    parser.add_argument(
        "--example_pool_path",
        type=str,
        default=None,
        help="Path to offline-built example pool JSONL.",
    )
    parser.add_argument("--dynamic_k", type=int, default=3, help="K: number of retrieved example items.")
    parser.add_argument("--dynamic_top_m", type=int, default=20, help="Top-M candidates before MMR selection.")
    parser.add_argument(
        "--dynamic_lambda_mmr",
        type=float,
        default=0.7,
        help="MMR lambda: similarity-vs-diversity tradeoff (0..1).",
    )
    parser.add_argument("--iteration_reasonsing_results_root_path", type=str,
                        default=None, help="Path to rule file")
    parser.add_argument("--model_name", type=str,
                        default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--is_zero", action="store_true",
                        help="Enable zero-shot rule generation")
    parser.add_argument("-k", type=int, default=0,
                        help="Number of generated rules, 0 denotes as much as possible")
    parser.add_argument("-f", type=int, default=5, help="Few-shot number")
    parser.add_argument("-topk", type=int, default=20, help="Top-k paths")
    parser.add_argument("-n", type=int, default=5, help="Number of threads")
    parser.add_argument("--test_one", type=str_to_bool, default='no',
                        help="Run only the first item of sampled_path and rule_head_with_zero for quick test")
    parser.add_argument("--test_common_num", type=int, default=10,
                        help="Test mode: number of common rule heads")
    parser.add_argument("--test_zero_num", type=int, default=20,
                        help="Test mode: number of zero rule heads")
    parser.add_argument("--test_iter_num", type=int, default=10,
                        help="Test mode: number of iteration heads")
    parser.add_argument("-l", type=int, default=3,
                        help="Sample times for generating k rules")
    parser.add_argument("--prefix", type=str, default="",
                        help="Prefix for files")
    parser.add_argument("--dry_run", action="store_true", help="Dry run mode")
    parser.add_argument("--is_rel_name", type=str_to_bool,
                        default='yes', help="Enable relation names")
    parser.add_argument("--select_with_confidence", type=str_to_bool,
                        default='no', help="Select with confidence")
    parser.add_argument('--clean_only', action='store_true',
                        help='Load summarized rules and clean rules only')
    parser.add_argument('--force_summarize',
                        action='store_true', help='Force summarize rules')
    parser.add_argument("--is_merge", type=str_to_bool,
                        default='no', help="Enable merge")
    parser.add_argument("--transition_distr", type=str,
                        default="exp", help="Transition distribution")
    parser.add_argument("--is_only_with_original_rules",
                        type=str_to_bool, default='no', help="Use only original rules")
    parser.add_argument("--is_high", type=str_to_bool,
                        default='No', help="Enable high mode")
    parser.add_argument("--min_conf", type=float,
                        default=0.01, help="Minimum confidence")
    parser.add_argument("--num_iter", type=int, default=2,
                        help="Number of iterations")
    parser.add_argument("-second", type=int, default=3,
                        help="Second sampling times for generating k rules")
    parser.add_argument("--bgkg", type=str, default="valid", choices=[
                        'train', 'train_valid', 'valid', 'test'], help="Background knowledge graph")
    parser.add_argument("--based_rule_type", type=str, default='low',
                        choices=['low', 'high'], help="Base rule type")
    parser.add_argument("--rule_domain", type=str, default='iteration',
                        choices=['iteration', 'high', 'all'], help="Rule domain")
    parser.add_argument("--bat_file_name", type=str, default='bat_file',
                        help="Batch file name")
    parser.add_argument("--results_root_path", type=str, default='results',
                        help="Results root path. Must put this parameter at last position on cmd line to avoid parsing error.")
    parser.add_argument("--is_generate_pathes_for_train_dataset", type=str_to_bool,
                        default='no', help="Enable generate_pathes_for_train_dataset for example pool construction")
    parser.add_argument("--selected_relations", type=str_to_bool,
                        default='no', help="Enable selected relations")
    parser.add_argument("--ex1_prompt", type=str_to_bool,
                    default='no', help="Enable selected relations")

    # Innovation-2: Relation Profile (semantic injection) for common stage prompt only.
    parser.add_argument(
        "--use_semantic_profile",
        type=str_to_bool,
        default=False,
        help="Enable semantic Relation Profile injection after candidate relations (common stage only).",
    )
    parser.add_argument(
        "--semantic_profile_path",
        type=str,
        default=None,
        help="Path to relation_profile.json. If None, uses results_root_path/bat_file_name/semantics/{dataset}/relation_profile.json",
    )
    parser.add_argument("--profile_use_country_pairs", type=str_to_bool, default=True, help="Ablation: include CountryPairs section.")
    parser.add_argument("--profile_use_time_stats", type=str_to_bool, default=True, help="Ablation: include TimeStats section.")
    parser.add_argument("--profile_use_repr_events", type=str_to_bool, default=True, help="Ablation: include RepresentativeEvents section.")
    parser.add_argument("--profile_topk_pairs", type=int, default=10, help="Max country pairs to show in prompt block.")
    parser.add_argument("--profile_events_k", type=int, default=3, help="Max representative events to show in prompt block.")
    parser.add_argument(
        "--clean_workers",
        type=int,
        default=1,
        help="0=auto, 1=serial, >1 parallel threads for clean stage",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=1,
        help="0=auto(cpu_count), 1=serial, >1 parallel processes for evaluation confidence",
    )
    parser.add_argument(
        "--eval_chunksize",
        type=int,
        default=8,
        help="multiprocessing imap chunksize for evaluation",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=1,
        help="0 disables reproducibility; non-zero enables per-rule deterministic seeding for confidence estimation",
    )
    parser.add_argument("--ex1_mode", default="mrr", type=str,
                    choices=['mrr', 'random', 'topk'])
    args, _ = parser.parse_known_args()
    return args, parser



def preload_tiktoken(model_name):
    # 用你真实用到的模型名
    enc = tiktoken.encoding_for_model(model_name)
    _ = enc.encode("warmup")


if __name__ == "__main__":
    args, parser = parse_arguments()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    preload_tiktoken(args.model_name)
    main(args, LLM)
