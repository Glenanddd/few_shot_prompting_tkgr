"""
Dynamic examples (few-shot) retrieval utilities for LLM-DA.

Design goals (MVP):
- Pure local computation (no online services / embeddings).
- Minimal-intrusion integration: only swap `examples` text when enabled.
- Stable + adjustable + ablation-friendly (k/top_m/lambda_mmr via CLI).

Retrieval signals:
- Interpretable relation-token TF-IDF features extracted from *sampled paths* (rule bodies).
- Diversity via MMR (Maximal Marginal Relevance) over the same TF-IDF space.

Prompt format constraint:
- Keep exactly `examples_title + examples_text` insertion style (same as prompt/common.json),
  to avoid breaking downstream LLM output parsing.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple


# Relation name tokenization for temporal atoms:
#   Return,_release_person(s)(X0,X1,T0)
# The relation name itself may contain parentheses "(s)", so we anchor on the *argument* pattern.
_REL_TOKEN_RE = re.compile(r"([\w\s'\-.,\(\)/]+)\(\w+,\s*\w+,\s*\w+\)")

# Some sources store rules as:
#   "<conf> <rule_supp> <body_supp>  <rule_core>"
# e.g. "0.004926     1   203  inv_Defend_verbally(X0,X2,T2)<-...".
# For prompt readability (and to match baseline common.json), we strip this prefix when formatting examples.
_METRIC_PREFIX_RE = re.compile(
    r"^\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?\s+\d+\s+\d+\s+"
)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_relation_tokens_from_rule(
    rule_text: str,
    *,
    from_body_only: bool = True,
) -> List[str]:
    """
    Extract relation-name tokens from a (temporal) rule string.

    By default we use ONLY the body-part (after '<-') to avoid trivial matching by the head relation name.
    This keeps the similarity signal focused on the sampled-path patterns.
    """
    text = rule_text or ""
    if from_body_only and "<-" in text:
        text = text.split("<-", 1)[1]
    tokens = [_normalize_ws(t) for t in _REL_TOKEN_RE.findall(text)]
    return [t for t in tokens if t]


def extract_relation_tokens_from_rules(
    rules: Sequence[str],
    *,
    from_body_only: bool = True,
) -> List[str]:
    tokens: List[str] = []
    for r in rules:
        tokens.extend(extract_relation_tokens_from_rule(r, from_body_only=from_body_only))
    return tokens


def _cosine_sim_sparse(
    vec_a: Dict[str, float], norm_a: float, vec_b: Dict[str, float], norm_b: float
) -> float:
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    # Iterate smaller dict for dot product.
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
        norm_a, norm_b = norm_b, norm_a
    dot = 0.0
    for key, val in vec_a.items():
        dot += val * vec_b.get(key, 0.0)
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class ExamplePoolIndex:
    """
    Precomputed TF-IDF vectors for pool items to make retrieval fast/stable.
    """

    items: Tuple[dict, ...]
    idf: Dict[str, float]
    vecs: Tuple[Dict[str, float], ...]
    norms: Tuple[float, ...]


def _build_tfidf_index(pool_items: Sequence[dict]) -> ExamplePoolIndex:
    # 1) Collect token TFs and document frequencies.
    tf_counters: List[Counter] = []
    df: Counter = Counter()
    for item in pool_items:
        # Prefer precomputed tokens (from example_pool_builder.py). Fallback: extract from paths text.
        tokens = item.get("path_rels")
        if not isinstance(tokens, list):
            paths = item.get("paths", []) or []
            tokens = extract_relation_tokens_from_rules(paths, from_body_only=True)
        tokens = [_normalize_ws(str(t)) for t in tokens if str(t).strip()]
        tf = Counter(tokens)
        tf_counters.append(tf)
        df.update(tf.keys())

    # 2) Compute IDF (smooth, stable).
    n_docs = max(1, len(pool_items))
    idf: Dict[str, float] = {}
    for token, doc_freq in df.items():
        idf[token] = math.log((n_docs + 1.0) / (doc_freq + 1.0)) + 1.0
    unknown_idf = math.log(n_docs + 1.0) + 1.0

    # 3) Build TF-IDF vectors + norms.
    vecs: List[Dict[str, float]] = []
    norms: List[float] = []
    for tf in tf_counters:
        vec = {t: float(c) * idf.get(t, unknown_idf) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        vecs.append(vec)
        norms.append(norm)

    return ExamplePoolIndex(
        items=tuple(pool_items),
        idf=idf,
        vecs=tuple(vecs),
        norms=tuple(norms),
    )


def load_example_pool_jsonl(example_pool_path: str) -> List[dict]:
    items: List[dict] = []
    with open(example_pool_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


@lru_cache(maxsize=4)
def load_example_pool_index(example_pool_path: str) -> ExamplePoolIndex:
    """
    Load + index the JSONL pool once per process (thread-safe via lru_cache internal lock).
    """
    if not os.path.exists(example_pool_path):
        raise FileNotFoundError(f"example_pool_path not found: {example_pool_path}")
    items = load_example_pool_jsonl(example_pool_path)
    if not items:
        raise ValueError(f"example_pool is empty: {example_pool_path}")
    return _build_tfidf_index(items)


def _build_query_vec(tokens: Sequence[str], idf: Dict[str, float], n_docs: int) -> Tuple[Dict[str, float], float]:
    tf = Counter(tokens)
    unknown_idf = math.log(n_docs + 1.0) + 1.0
    vec = {t: float(c) * idf.get(t, unknown_idf) for t, c in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return vec, norm


def retrieve_dynamic_examples(
    sampled_paths: List[str],
    pool_items,
    k: int,
    top_m: int,
    lambda_mmr: float,
    mode: str = "mmr",          # "mmr" | "topk" | "random"
    random_seed: int = 0,
) -> List[dict]:
    """
    Retrieve k pool items for the current head relation's sampled_paths.

    Modes:
      - "mmr": Similarity + MMR (Full method)
      - "topk": Similarity only (Ablation-1)
      - "random": Random dynamic examples (Ablation-2)

    Notes:
      - Randomness is ONLY used in mode="random".
      - random_seed is fixed to ensure reproducibility for ablation.
    """
    if k <= 0:
        return []

    # Build or use TF-IDF index
    if isinstance(pool_items, ExamplePoolIndex):
        index = pool_items
    else:
        index = _build_tfidf_index(pool_items)

    if not index.items:
        return []

    # ---------- Ablation-2: Random Dynamic Examples ----------
    if mode == "random":
        rng = random.Random(int(random_seed))
        items = list(index.items)

        if top_m is not None:
            m = max(1, min(int(top_m), len(items)))
            items = items[:m]  # stable prefix

        if k >= len(items):
            return items

        return rng.sample(items, k)

    # ---------- Similarity-based modes ----------
    # Extract relation-name tokens from sampled paths (body only)
    query_tokens = extract_relation_tokens_from_rules(
        sampled_paths, from_body_only=True
    )
    query_tokens = [_normalize_ws(t) for t in query_tokens if t]

    if not query_tokens:
        # Edge-case fallback: deterministic
        return list(index.items[: min(k, len(index.items))])

    # Build query vector
    query_vec, query_norm = _build_query_vec(
        query_tokens, index.idf, n_docs=len(index.items)
    )

    # Compute similarity(query, item)
    sim_q = [
        _cosine_sim_sparse(
            query_vec, query_norm, index.vecs[i], index.norms[i]
        )
        for i in range(len(index.items))
    ]

    # Candidate pool: top_m by similarity
    m = len(index.items) if top_m is None else max(
        1, min(int(top_m), len(index.items))
    )
    candidate_indices = sorted(
        range(len(index.items)),
        key=lambda i: sim_q[i],
        reverse=True
    )[:m]

    # ---------- Ablation-1: Only Similarity (Top-K) ----------
    if mode == "topk" or lambda_mmr >= 1.0:
        topk_idx = candidate_indices[: min(k, len(candidate_indices))]
        return [index.items[i] for i in topk_idx]

    # ---------- Full Method: Similarity + MMR ----------
    selected = []
    selected_set = set()

    lambda_mmr = float(lambda_mmr)
    lambda_mmr = max(0.0, min(1.0, lambda_mmr))

    while len(selected) < min(k, len(candidate_indices)):
        if not selected:
            best = max(candidate_indices, key=lambda i: sim_q[i])
            selected.append(best)
            selected_set.add(best)
            continue

        best_idx = None
        best_score = -1e18

        for cand in candidate_indices:
            if cand in selected_set:
                continue

            # Diversity term
            max_sim_selected = 0.0
            for s in selected:
                sim_cs = _cosine_sim_sparse(
                    index.vecs[cand], index.norms[cand],
                    index.vecs[s], index.norms[s]
                )
                if sim_cs > max_sim_selected:
                    max_sim_selected = sim_cs

            score = (
                lambda_mmr * sim_q[cand]
                - (1.0 - lambda_mmr) * max_sim_selected
            )

            if score > best_score:
                best_score = score
                best_idx = cand

        if best_idx is None:
            break

        selected.append(best_idx)
        selected_set.add(best_idx)

    return [index.items[i] for i in selected]



def format_examples(items: Sequence[dict]) -> str:
    """
    Format retrieved items into a single `examples` string compatible with prompt/common.json.

    IMPORTANT: keep the same section labels and plain-text structure to avoid breaking downstream parsing.
    """
    def _strip_leading_metrics(line: str) -> str:
        line = str(line).rstrip("\n")
        return _METRIC_PREFIX_RE.sub("", line).strip()

    blocks: List[str] = []
    for item in items:
        head_name = item.get("head_rel_name", "")
        paths = item.get("paths", []) or []
        rules = item.get("rules", []) or []
        lines: List[str] = [
            f"Rule head: {head_name}(X0,Xl,Tl)",
            "Sampled rules:",
        ]
        lines.extend([_strip_leading_metrics(p) for p in paths if str(p).strip()])
        lines.append("Generated Temporal logic rules:")
        lines.extend([_strip_leading_metrics(r) for r in rules if str(r).strip()])
        blocks.append("\n".join(lines))
    return "\n".join(blocks).strip()


def format_examples_ex1(items: Sequence[dict]) -> str:
    """
    Format retrieved items into a single `examples` string compatible with prompt/common.json.

    IMPORTANT: keep the same section labels and plain-text structure to avoid breaking downstream parsing.
    """
    def _strip_leading_metrics(line: str) -> str:
        line = str(line).rstrip("\n")
        return _METRIC_PREFIX_RE.sub("", line).strip()

    blocks: List[str] = []
    for item in items:
        head_name = item.get("head_rel_name", "")
        paths = item.get("paths", []) or []
        rules = item.get("rules", []) or []
        lines: List[str] = [
            f"Rule Head: {head_name}(X0,Xl,Tl)",
            "Input: Sampled Random Walk Paths from the Temporal Knowledge Graph:",
        ]
        lines.extend([_strip_leading_metrics(p) for p in paths if str(p).strip()])
        lines.append("Output: Generated Temporal logic rules:")
        lines.extend([_strip_leading_metrics(r) for r in rules if str(r).strip()])
        blocks.append("\n".join(lines))
    return "\n".join(blocks).strip()

def _selfcheck_dynamic_examples() -> None:
    # Fake pool items with partially overlapping body relations to test similarity+MMR.
    pool_items = [
        {
            "ex_id": "R1#0",
            "head_rel_name": "R1",
            "paths": ["R1(X0,X1,T1)<-A(X0,X1,T0)&B(X1,X2,T0)\n"],
            "rules": ["R1(X0,X1,T1)<-A(X0,X1,T0)\n"],
            "path_rels": ["A", "B"],
        },
        {
            "ex_id": "R2#0",
            "head_rel_name": "R2",
            "paths": ["R2(X0,X1,T1)<-A(X0,X1,T0)&C(X1,X2,T0)\n"],
            "rules": ["R2(X0,X1,T1)<-C(X0,X1,T0)\n"],
            "path_rels": ["A", "C"],
        },
        {
            "ex_id": "R3#0",
            "head_rel_name": "R3",
            "paths": ["R3(X0,X1,T1)<-D(X0,X1,T0)&E(X1,X2,T0)\n"],
            "rules": ["R3(X0,X1,T1)<-D(X0,X1,T0)\n"],
            "path_rels": ["D", "E"],
        },
    ]
    index = _build_tfidf_index(pool_items)
    sampled_paths = ["Target(X0,X1,T1)<-A(X0,X1,T0)&B(X1,X2,T0)\n"]
    selected = retrieve_dynamic_examples(sampled_paths, index, k=2, top_m=3, lambda_mmr=0.7)
    assert len(selected) == 2, f"expected k=2, got {len(selected)}"
    text = format_examples(selected)
    assert "Rule head:" in text
    assert "Generated Temporal logic rules:" in text
    print("[OK] dynamic examples self-check passed.")
    print(text)


if __name__ == "__main__":
    _selfcheck_dynamic_examples()
