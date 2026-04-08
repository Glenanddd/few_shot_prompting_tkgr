import argparse
import os
import shutil

import copy
import sys
import torch
import numpy as np
import time
import threading
import multiprocessing as mp

from grapher import Grapher
from rule_learning import Rule_Learner, rules_statistics, rule_key, verbalize_rule
from temporal_walk import initialize_temporal_walk
from utils_windows_long_path import maybe_windows_long_path

from joblib import Parallel, delayed
from datetime import datetime
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 设置HF_ENDPOINT环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from params import str_to_bool

from utils import load_json_data, save_json_data


def select_similary_relations(relation2id, output_dir):
    id2relation = dict([(v, k) for k, v in relation2id.items()])

    save_json_data(id2relation, maybe_windows_long_path(os.path.join(output_dir, 'transfomers_id2rel.json')))
    save_json_data(relation2id, maybe_windows_long_path(os.path.join(output_dir, 'transfomers_rel2id.json')))

    all_rels = list(relation2id.keys())
    # 加载预训练的模型
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # 定义句子
    sentences_A = all_rels
    sentences_B = all_rels

    # 使用模型为句子编码
    embeddings_A = model.encode(sentences_A)
    embeddings_B = model.encode(sentences_B)

    # 计算句子之间的余弦相似度
    similarity_matrix = cosine_similarity(embeddings_A, embeddings_B)

    np.fill_diagonal(similarity_matrix, 0)

    np.save(maybe_windows_long_path(os.path.join(output_dir, 'matrix.npy')), similarity_matrix)


def build_confidence_concrete(rules_dict, concrete_map, id2relation, id2entity, id2ts):
    """
    Enrich confidence.json with human-readable names, abstract rule string,
    and a list of concrete walks (entities + timestamps) that generated the rule.
    """
    output = {}
    for head_rel, rules in rules_dict.items():
        head_key = str(head_rel)
        output.setdefault(head_key, [])
        for rule in rules:
            # Copy original fields
            rule_entry = copy.deepcopy(rule)
            rule_entry["head_rel_name"] = id2relation[rule["head_rel"]]
            rule_entry["body_rels_names"] = [id2relation[x] for x in rule["body_rels"]]
            rule_entry["abstract_rule"] = verbalize_rule(copy.deepcopy(rule), id2relation)
            concrete_list = sorted(
                list(concrete_map.get(rule_key(rule), set()))
            )
            rule_entry["concrete_rule"] = concrete_list
            output[head_key].append(rule_entry)
    return output

def format_hms(seconds):
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def main(parsed):
    
    parsed["results_root_path"] = parsed["results_root_path"].strip('"')
    parsed["output_path"] = maybe_windows_long_path(os.path.join(parsed["results_root_path"], parsed["bat_file_name"], 'sampled_path'))

    
    dataset = parsed["dataset"]
    rule_lengths = parsed["max_path_len"]
    rule_lengths = (torch.arange(rule_lengths) + 1).tolist()
    num_walks = parsed["num_walks"]
    transition_distr = parsed["transition_distr"]
    num_processes = parsed["cores"]
    seed = parsed["seed"]
    version = parsed["version"]
    

    dataset_dir = maybe_windows_long_path(os.path.join("datasets", dataset))
    data = Grapher(dataset_dir)

    temporal_walk = initialize_temporal_walk(version, data, transition_distr)

    rl = Rule_Learner(
        temporal_walk.edges,
        data.id2relation,
        data.inv_relation_id,
        dataset,
        id2entity=data.id2entity,
        id2ts=data.id2ts,
        version=version,
        output_path=parsed["output_path"]
    )
    all_relations = sorted(temporal_walk.edges)  # Learn for all relations
    all_relations = [int(item) for item in all_relations]
    rel2idx = data.relation2id

    select_similary_relations(data.relation2id, rl.output_dir)

    constant_config = load_json_data(maybe_windows_long_path(os.path.join("Config", "constant.json")))
    relation_regex = constant_config['relation_regex'][dataset]

    total_tasks = len(all_relations) * len(rule_lengths)
    progress_counter = None
    progress_lock = None
    if total_tasks > 0:
        manager = mp.Manager()
        progress_counter = manager.Value("i", 0)
        progress_lock = manager.Lock()

    def learn_rules(i, num_relations, use_relax_time=False):
        """
        Learn rules with optional relax time (multiprocessing possible).

        Parameters:
            i (int): process number
            num_relations (int): minimum number of relations for each process
            use_relax_time (bool): Whether to use relax time in sampling

        Returns:
            rl.rules_dict (dict): rules dictionary
        """

        set_seed_if_provided()
        relations_idx = calculate_relations_idx(i, num_relations)
        num_rules = [0]

        for k in relations_idx:
            rel = all_relations[k]
            for length in rule_lengths:
                it_start = time.time()
                process_rules_for_relation(rel, length, use_relax_time)
                it_end = time.time()
                it_time = round(it_end - it_start, 6)
                num_rules.append(sum([len(v) for k, v in rl.rules_dict.items()]) // 2)
                num_new_rules = num_rules[-1] - num_rules[-2]

                # print(f"Process {i}: relation {k - relations_idx[0] + 1}/{len(relations_idx)}, length {length}: {it_time} sec, {num_new_rules} rules")
                if progress_counter is not None:
                    with progress_lock:
                        progress_counter.value = progress_counter.value + 1

        return rl.rules_dict, rl.concrete_map

    def set_seed_if_provided():
        if seed:
            np.random.seed(seed)

    def calculate_relations_idx(i, num_relations):
        if i < num_processes - 1:
            return range(i * num_relations, (i + 1) * num_relations)
        else:
            return range(i * num_relations, len(all_relations))

    def process_rules_for_relation(rel, length, use_relax_time):
        for _ in range(num_walks):
            walk_successful, walk = temporal_walk.sample_walk(length + 1, rel, use_relax_time)
            if walk_successful:
                rl.create_rule(walk, use_relax_time)

    start = time.time()
    progress_stop = threading.Event()

    def progress_monitor():
        if total_tasks <= 0:
            return
        if tqdm is not None:
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            with tqdm(
                total=total_tasks,
                desc="Total progress",
                unit="task",
                dynamic_ncols=True,
                bar_format=bar_format,
                file=sys.stdout,
            ) as pbar:
                last = 0
                while not progress_stop.is_set():
                    with progress_lock:
                        current = progress_counter.value
                    if current > last:
                        pbar.update(current - last)
                        last = current
                    if last >= total_tasks:
                        break
                    time.sleep(0.2)
                if last < total_tasks:
                    pbar.update(total_tasks - last)
        else:
            last = -1
            while not progress_stop.is_set():
                with progress_lock:
                    current = progress_counter.value
                if current != last:
                    elapsed = time.time() - start
                    print(
                        f"Total progress: {current}/{total_tasks} | elapsed {format_hms(elapsed)}",
                        end="\r",
                        flush=True,
                    )
                    last = current
                if current >= total_tasks:
                    break
                time.sleep(0.5)
            print()

    progress_thread = None
    if progress_counter is not None:
        progress_thread = threading.Thread(target=progress_monitor, daemon=True)
        progress_thread.start()
    num_relations = len(all_relations) // num_processes
    try:
        output = Parallel(n_jobs=num_processes)(
            delayed(learn_rules)(i, num_relations, parsed['is_relax_time']) for i in range(num_processes)
        )
    finally:
        if progress_thread is not None:
            progress_stop.set()
            progress_thread.join()
    end = time.time()
    all_graph, all_concrete = output[0]
    for i in range(1, num_processes):
        all_graph.update(output[i][0])
        # merge concrete maps (sets)
        for k, v in output[i][1].items():
            if k not in all_concrete:
                all_concrete[k] = v
            else:
                all_concrete[k].update(v)

    total_time = round(end - start, 6)
    print("Learning finished in {} seconds ({}).".format(total_time, format_hms(total_time)))

    rl.rules_dict = all_graph
    rl.concrete_map = all_concrete
    rl.sort_rules_dict()
    dt = datetime.now()
    dt = dt.strftime("%d%m%y%H%M%S")
    rl.save_rules(dt, rule_lengths, num_walks, transition_distr, seed)
    save_json_data(rl.rules_dict, maybe_windows_long_path(os.path.join(rl.output_dir, 'confidence.json')))
    confidence_concrete = build_confidence_concrete(
        rl.rules_dict, rl.concrete_map, data.id2relation, data.id2entity, data.id2ts
    )
    save_json_data(confidence_concrete, maybe_windows_long_path(os.path.join(rl.output_dir, 'confidence_concrete.json')))
    rules_statistics(rl.rules_dict)
    rl.save_rules_verbalized(dt, rule_lengths, num_walks, transition_distr, seed, rel2idx, relation_regex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets', help='data directory')
    parser.add_argument("--dataset", "-d", default="", type=str)
    parser.add_argument("--max_path_len", "-m", type=int, default=3, help="max sampled path length")
    parser.add_argument("--anchor", type=int, default=5, help="anchor facts for each relation")
    parser.add_argument("--output_path", type=str, default="sampled_path", help="output path")
    parser.add_argument("--sparsity", type=float, default=1, help="dataset sampling sparsity")
    parser.add_argument("--cores", "-p", type=int, default=5, help="dataset sampling sparsity")
    parser.add_argument("--num_walks", "-n", default="100", type=int)
    parser.add_argument("--transition_distr", default="exp", type=str)
    parser.add_argument("--seed", "-s", default=None, type=int)
    parser.add_argument("--window", "-w", default=0, type=int)
    parser.add_argument("--version", default="train", type=str,
                        choices=['train', 'test', 'train_valid', 'valid'])
    parser.add_argument("--is_relax_time", default='no', type=str_to_bool)
    parser.add_argument("--bat_file_name", type=str, default='bat_file',
                        help="Batch file name")
    parser.add_argument("--results_root_path", type=str, default='results',
                        help="Results root path. Must put this parameter at last position on cmd line to avoid parsing error.")

    parsed = vars(parser.parse_args())

    main(parsed)
