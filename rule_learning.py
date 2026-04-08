import os
import json
import itertools
import shutil
import numpy as np
from collections import Counter, defaultdict

import copy
import re
import traceback

from utils import save_json_data, write_to_file
from utils_windows_long_path import maybe_windows_long_path


def normalize_var_constraints(var_constraints, body_len):
    """
    Normalize variable constraints by adding singleton positions for missing indices.
    This keeps key construction stable and avoids mutating the original list.
    """
    if var_constraints:
        vc = [sorted(list(x)) for x in copy.deepcopy(var_constraints)]
        flat = [idx for grp in vc for idx in grp]
        for i in range(body_len + 1):
            if i not in flat:
                vc.append([i])
        vc = [tuple(x) for x in vc]
        return tuple(sorted(vc))
    return tuple((i,) for i in range(body_len + 1))


def rule_key(rule):
    """Hashable key for an abstract rule."""
    return (
        rule["head_rel"],
        tuple(rule["body_rels"]),
        normalize_var_constraints(rule["var_constraints"], len(rule["body_rels"])),
    )


class Rule_Learner(object):
    def __init__(self, edges, id2relation, inv_relation_id, dataset, id2entity=None, id2ts=None, version=None, output_path=None):
        """
        Initialize rule learner object.

        Parameters:
            edges (dict): edges for each relation
            id2relation (dict): mapping of index to relation
            inv_relation_id (dict): mapping of relation to inverse relation
            dataset (str): dataset name

        Returns:
            None
        """

        self.edges = edges
        # === 加速索引：不影响随机流/逻辑 ===
        self.edges_by_src = self._build_edges_by_src_index(self.edges)  # P2: rel+src -> 行索引
        self._head_pair_max_ts_cache = {}  # P1: head_rel 缓存 (s,o)->max_ts

        self.id2relation = id2relation
        self.inv_relation_id = inv_relation_id
        self.num_individual = 0
        self.num_shared = 0
        self.num_original = 0

        self.found_rules = []
        self.rule2confidence_dict = {}
        self.original_found_rules = []
        self.rules_dict = dict()
        self.concrete_map = defaultdict(set)
        self.id2entity = id2entity
        self.id2ts = id2ts

        if output_path is not None:
            if version == 'valid':
                self.output_dir = maybe_windows_long_path(os.path.join(output_path, f"{dataset}_valid"))
            else:
                self.output_dir = maybe_windows_long_path(os.path.join(output_path, f"{dataset}"))
                
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir, ignore_errors=False)
            os.makedirs(self.output_dir, exist_ok=True)
            

    def _pair_key_32(self, s, o):
        """
        将 (s,o) 打包成一个 int64 key： (s<<32) | o
        前提：实体 id 在 0..2^32-1 范围内（ICEWS/常见KG都满足）。
        """
        return (int(s) << 32) | int(o)

    def _build_edges_by_src_index(self, edges_dict):
        """
        为每个 rel 建立：src -> 行索引数组（索引指向 edges_dict[rel] 的行）
        关键要求：同一个 src 的行索引顺序必须与原 edges[rel] 中出现顺序一致，
        才能保证 np.random.choice 选到的边与原实现完全一致。
        """
        edges_by_src = {}
        for rel, arr in edges_dict.items():
            if arr is None or len(arr) == 0:
                edges_by_src[rel] = {}
                continue

            srcs = arr[:, 0].astype(np.int64, copy=False)
            # mergesort 是稳定排序：相同 src 的相对顺序保持为原数组顺序（非常关键）
            order = np.argsort(srcs, kind="mergesort")
            srcs_sorted = srcs[order]

            # 找每个 src 分组边界
            # group_starts: 每个新 src 的起点
            diff = np.diff(srcs_sorted)
            group_starts = np.concatenate(([0], np.nonzero(diff)[0] + 1))
            group_ends = np.concatenate((group_starts[1:], [len(order)]))

            d = {}
            for st, ed in zip(group_starts, group_ends):
                s_val = int(srcs_sorted[st])
                d[s_val] = order[st:ed]  # numpy array of row indices, 顺序与原实现一致
            edges_by_src[rel] = d

        return edges_by_src

    def _get_head_pair_max_ts(self, head_rel):
        """
        懒加载缓存：对某个 head_rel，构建 (s,o)->max_ts 的 dict。
        用于 P1：calculate_rule_support 从扫全表变成 O(1) 查询。
        """
        cache = self._head_pair_max_ts_cache
        if head_rel in cache:
            return cache[head_rel]

        head_edges = self.edges.get(head_rel, None)
        if head_edges is None or len(head_edges) == 0:
            cache[head_rel] = None
            return None

        s = head_edges[:, 0].astype(np.int64, copy=False)
        o = head_edges[:, 2].astype(np.int64, copy=False)
        t = head_edges[:, 3].astype(np.int64, copy=False)

        keys = (s << 32) | o
        order = np.argsort(keys, kind="mergesort")
        keys_sorted = keys[order]
        t_sorted = t[order]

        # 按 key 分组并取每组 max(ts)
        diff = np.diff(keys_sorted)
        group_starts = np.concatenate(([0], np.nonzero(diff)[0] + 1))
        group_ends = np.concatenate((group_starts[1:], [len(keys_sorted)]))

        # 这里用 Python 循环建 dict：组数=unique (s,o) 数，通常远小于边数
        pair2maxts = {}
        for st, ed in zip(group_starts, group_ends):
            k = int(keys_sorted[st])
            # 取该组最大时间戳
            # t_sorted[st:ed] 是该 pair 的所有时间
            pair2maxts[k] = int(t_sorted[st:ed].max())

        cache[head_rel] = pair2maxts
        return pair2maxts


    def create_rule(self, walk, confidence=0, use_relax_time=False):
        """
        Create a rule given a cyclic temporal random walk.
        The rule contains information about head relation, body relations,
        variable constraints, confidence, rule support, and body support.
        A rule is a dictionary with the content
        {"head_rel": int, "body_rels": list, "var_constraints": list,
         "conf": float, "rule_supp": int, "body_supp": int}

        Parameters:
            walk (dict): cyclic temporal random walk
                         {"entities": list, "relations": list, "timestamps": list}
            confidence (float): confidence value
            use_relax_time (bool): whether the rule is created with relaxed time

        Returns:
            rule (dict): created rule
        """

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        self.record_concrete_walk(rule, walk)
        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            (
                rule["conf"],
                rule["rule_supp"],
                rule["body_supp"],
            ) = self.estimate_confidence(rule, is_relax_time=use_relax_time)

            rule["llm_confidence"] = confidence

            if rule["conf"] or confidence:
                self.update_rules_dict(rule)

    def create_rule_for_merge(self, walk, confidence=0, rule_without_confidence="", rules_var_dict=None,
                              is_merge=False, is_relax_time=False):
        """
        Create a rule given a cyclic temporal random walk.
        The rule contains information about head relation, body relations,
        variable constraints, confidence, rule support, and body support.
        A rule is a dictionary with the content
        {"head_rel": int, "body_rels": list, "var_constraints": list,
         "conf": float, "rule_supp": int, "body_supp": int}

        Parameters:
            walk (dict): cyclic temporal random walk
                         {"entities": list, "relations": list, "timestamps": list}

        Returns:
            rule (dict): created rule
        """

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        self.record_concrete_walk(rule, walk)
        if is_merge is True:
            if rules_var_dict.get(rule_without_confidence) is None:
                if rule not in self.found_rules:
                    self.found_rules.append(rule.copy())
                    (
                        rule["conf"],
                        rule["rule_supp"],
                        rule["body_supp"],
                    ) = self.estimate_confidence(rule)

                    rule["llm_confidence"] = confidence

                    if rule["conf"] or confidence:
                        self.num_individual += 1
                        self.update_rules_dict(rule)


            else:
                rule_var = rules_var_dict[rule_without_confidence]
                rule_var["llm_confidence"] = confidence
                temp_var = {}
                temp_var['head_rel'] = rule_var['head_rel']
                temp_var['body_rels'] = rule_var['body_rels']
                temp_var["var_constraints"] = rule_var["var_constraints"]
                if temp_var not in self.original_found_rules:
                    self.original_found_rules.append(temp_var.copy())
                    self.update_rules_dict(rule_var)
                    self.num_shared += 1
        else:
            if rule not in self.found_rules:
                self.found_rules.append(rule.copy())
                (
                    rule["conf"],
                    rule["rule_supp"],
                    rule["body_supp"],
                ) = self.estimate_confidence(rule, is_relax_time=is_relax_time)

                # if rule["body_supp"] == 0:
                #     rule["body_supp"] = 2

                rule["llm_confidence"] = confidence

                if rule["conf"] or confidence:
                    self.update_rules_dict(rule)

    def create_rule_for_merge_for_iteration(self, walk, llm_confidence=0, rule_without_confidence="",
                                            rules_var_dict=None,
                                            is_merge=False):
        """
        Create a rule given a cyclic temporal random walk.
        The rule contains information about head relation, body relations,
        variable constraints, confidence, rule support, and body support.
        A rule is a dictionary with the content
        {"head_rel": int, "body_rels": list, "var_constraints": list,
         "conf": float, "rule_supp": int, "body_supp": int}

        Parameters:
            walk (dict): cyclic temporal random walk
                         {"entities": list, "relations": list, "timestamps": list}

        Returns:
            rule (dict): created rule
        """

        rule = dict()
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            self.inv_relation_id[x] for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        self.record_concrete_walk(rule, walk)
        rule_with_confidence = ""

        if is_merge is True:
            if rules_var_dict.get(rule_without_confidence) is None:
                if rule not in self.found_rules:
                    self.found_rules.append(rule.copy())
                    (
                        rule["conf"],
                        rule["rule_supp"],
                        rule["body_supp"],
                    ) = self.estimate_confidence(rule)

                    tuple_key = str(rule)
                    self.rule2confidence_dict[tuple_key] = rule["conf"]
                    rule_with_confidence = rule_without_confidence + '&' + str(rule["conf"])

                    rule["llm_confidence"] = llm_confidence

                    if rule["conf"] or llm_confidence:
                        self.num_individual += 1
                        self.update_rules_dict(rule)
                else:
                    tuple_key = tuple(rule)
                    confidence = self.rule2confidence_dict[tuple_key]
                    rule_with_confidence = rule_without_confidence + '&' + confidence


            else:
                rule_var = rules_var_dict[rule_without_confidence]
                rule_var["llm_confidence"] = llm_confidence
                temp_var = {}
                temp_var['head_rel'] = rule_var['head_rel']
                temp_var['body_rels'] = rule_var['body_rels']
                temp_var["var_constraints"] = rule_var["var_constraints"]
                if temp_var not in self.original_found_rules:
                    self.original_found_rules.append(temp_var.copy())
                    self.update_rules_dict(rule_var)
                    self.num_shared += 1
        else:
            if rule not in self.found_rules:
                tuple_key = str(rule)
                self.found_rules.append(rule.copy())
                (
                    rule["conf"],
                    rule["rule_supp"],
                    rule["body_supp"],
                ) = self.estimate_confidence(rule)

                self.rule2confidence_dict[tuple_key] = rule["conf"]
                rule_with_confidence = rule_without_confidence + '&' + str(rule["conf"])

                if rule["body_supp"] == 0:
                    rule["body_supp"] = 2

                rule["llm_confidence"] = llm_confidence

                if rule["conf"] or llm_confidence:
                    self.update_rules_dict(rule)
            else:
                tuple_key = str(rule)
                confidence = self.rule2confidence_dict[tuple_key]
                rule_with_confidence = rule_without_confidence + '&' + str(confidence)

        return rule_with_confidence

    def record_concrete_walk(self, rule, walk):
        """
        Record the concrete walk (entities/timestamps) that produced an abstract rule.
        Multiple concrete walks for the same abstract rule are stored without duplication.
        """
        if (self.id2entity is None) or (self.id2ts is None):
            return
        try:
            concrete = verbalize_concrete_rule(
                rule, walk, self.id2relation, self.id2entity, self.id2ts
            )
        except Exception:
            # Robust to any formatting errors; do not block rule learning.
            return
        key = rule_key(rule)
        self.concrete_map[key].add(concrete)

    def define_var_constraints(self, entities):
        """
        Define variable constraints, i.e., state the indices of reoccurring entities in a walk.

        Parameters:
            entities (list): entities in the temporal walk

        Returns:
            var_constraints (list): list of indices for reoccurring entities
        """

        var_constraints = []
        for ent in set(entities):
            all_idx = [idx for idx, x in enumerate(entities) if x == ent]
            var_constraints.append(all_idx)
        var_constraints = [x for x in var_constraints if len(x) > 1]

        return sorted(var_constraints)


    def estimate_confidence(self, rule, num_samples=2000, is_relax_time=False):
        """
        Estimate the confidence of the rule by sampling bodies and checking the rule support.
        （加速版：set 去重替代 sort+groupby；不改变随机流/逻辑）
        """
        if any(body_rel not in self.edges for body_rel in rule["body_rels"]):
            return 0, 0, 0

        if rule["head_rel"] not in self.edges:
            return 0, 0, 0

        # 用 set 去重（P3）
        unique_set = set()

        for _ in range(num_samples):
            sample_successful, body_ents_tss = self.sample_body(
                rule["body_rels"], rule["var_constraints"], is_relax_time
            )
            if sample_successful:
                # 转成 tuple 才能放进 set；内容不变
                unique_set.add(tuple(body_ents_tss))

        body_support = len(unique_set)

        confidence, rule_support = 0, 0
        if body_support:
            # 为了保持确定性（可选），排序后再遍历；即使不排序，计数也不受影响
            unique_bodies = sorted(unique_set)
            rule_support = self.calculate_rule_support(unique_bodies, rule["head_rel"])
            confidence = round(rule_support / body_support, 6)

        return confidence, rule_support, body_support



    def sample_body(self, body_rels, var_constraints, use_relax_time=False):
        """
        Sample a walk according to the rule body.
        The sequence of timesteps should be non-decreasing.
        （加速版：用 rel+src 索引减少全表 mask 扫描；不改变随机流/逻辑）
        """
        sample_successful = True
        body_ents_tss = []

        edges = self.edges
        edges_by_src = self.edges_by_src
        choice = np.random.choice  # 不改变随机流（仍然每次只调用一次 choice）

        cur_rel = body_rels[0]
        rel_edges = edges[cur_rel]
        next_edge = rel_edges[choice(len(rel_edges))]  # 与原实现一致
        cur_ts = next_edge[3]
        cur_node = next_edge[2]

        body_ents_tss.append(next_edge[0])
        body_ents_tss.append(cur_ts)
        body_ents_tss.append(cur_node)

        for cur_rel in body_rels[1:]:
            next_edges = edges[cur_rel]

            src = int(cur_node)
            idxs = edges_by_src[cur_rel].get(src, None)
            if idxs is None or len(idxs) == 0:
                sample_successful = False
                break

            # 先按 src 取子集（顺序与原 mask 保持一致）
            edges_src = next_edges[idxs]

            if use_relax_time:
                filtered_edges = edges_src
            else:
                # 再按时间过滤（顺序仍保持一致）
                mask_time = (edges_src[:, 3] >= cur_ts)
                filtered_edges = edges_src[mask_time]

            if len(filtered_edges):
                next_edge = filtered_edges[choice(len(filtered_edges))]  # 与原实现一致
                cur_ts = next_edge[3]
                cur_node = next_edge[2]
                body_ents_tss.append(cur_ts)
                body_ents_tss.append(cur_node)
            else:
                sample_successful = False
                break

        if sample_successful and var_constraints:
            # Check variable constraints（不变）
            body_var_constraints = self.define_var_constraints(body_ents_tss[::2])
            if body_var_constraints != var_constraints:
                sample_successful = False

        return sample_successful, body_ents_tss

    def calculate_rule_support(self, unique_bodies, head_rel):
        """
        Calculate the rule support.
        （加速版：用 (s,o)->max_ts 索引避免每个 body 扫 head_rel 全表；逻辑等价）
        """
        rule_support = 0

        pair2maxts = self._get_head_pair_max_ts(head_rel)
        if not pair2maxts:
            return 0

        for body in unique_bodies:
            # body: [s, ts1, e1, ts2, e2, ... , last_ts, o] 或 tuple 同结构
            s = body[0]
            o = body[-1]
            last_ts = body[-2]

            k = self._pair_key_32(s, o)
            max_ts = pair2maxts.get(k, None)
            if max_ts is not None and max_ts > last_ts:  # 严格大于（与原实现一致）
                rule_support += 1

        return rule_support


    def update_rules_dict(self, rule):
        """
        Update the rules if a new rule has been found.

        Parameters:
            rule (dict): generated rule from self.create_rule

        Returns:
            None
        """

        try:
            self.rules_dict[rule["head_rel"]].append(rule)
        except KeyError:
            self.rules_dict[rule["head_rel"]] = [rule]

    def sort_rules_dict(self):
        """
        Sort the found rules for each head relation by decreasing confidence.

        Parameters:
            None

        Returns:
            None
        """

        for rel in self.rules_dict:
            self.rules_dict[rel] = sorted(
                self.rules_dict[rel], key=lambda x: x["conf"], reverse=True
            )

    def save_rules(self, dt, rule_lengths, num_walks, transition_distr, seed):
        """
        Save all rules.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """

        rules_dict = {int(k): v for k, v in self.rules_dict.items()}
        filename = "{0}_r{1}_n{2}_{3}_s{4}_rules.json".format(
            dt, rule_lengths, num_walks, transition_distr, seed
        )
        filename = filename.replace(" ", "")
        with open(maybe_windows_long_path(os.path.join(self.output_dir, filename)), "w", encoding="utf-8") as fout:
            json.dump(rules_dict, fout)

    def save_rules_verbalized(self, dt, rule_lengths, num_walks, transition_distr, seed, rel2idx, relation_regex):
        """
        Save all rules in a human-readable format.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """

        output_original_dir = maybe_windows_long_path(os.path.join(self.output_dir, 'original'))
        os.makedirs(output_original_dir, exist_ok=True)

        rules_str, rules_var = self.verbalize_rules()
        save_json_data(rules_var, maybe_windows_long_path(os.path.join(output_original_dir, "rules_var.json")))
        filename = self.generate_filename(dt, rule_lengths, num_walks, transition_distr, seed, "rules.txt")
        
        write_to_file(rules_str, maybe_windows_long_path(os.path.join(self.output_dir, filename)))

        original_rule_txt = maybe_windows_long_path(os.path.join(self.output_dir, filename))
        remove_filename = self.generate_filename(dt, rule_lengths, num_walks, transition_distr, seed,
                                                 "remove_rules.txt")
        
        rule_id_content = self.remove_first_three_columns(maybe_windows_long_path(os.path.join(self.output_dir, filename)),
                                                          maybe_windows_long_path(os.path.join(self.output_dir, "rules.txt")))
        
        rule_id_content = self.remove_first_three_columns(maybe_windows_long_path(os.path.join(self.output_dir, filename)),
                                                          maybe_windows_long_path(os.path.join(self.output_dir, remove_filename)))

        self.parse_and_save_rules(remove_filename, list(rel2idx.keys()), relation_regex, 'closed_rel_paths.jsonl')
        self.parse_and_save_rules_with_names(remove_filename, rel2idx, relation_regex, 'rules_name.json',
                                             rule_id_content)
        self.parse_and_save_rules_with_ids(rule_id_content, rel2idx, relation_regex, 'rules_id.json')

        self.save_rule_name_with_confidence(original_rule_txt, relation_regex,
                                       maybe_windows_long_path(os.path.join(self.output_dir, 'relation_name_with_confidence.json')), list(rel2idx.keys()))

    def verbalize_rules(self):
        rules_str = ""
        rules_var = {}
        for rel in self.rules_dict:
            for rule in self.rules_dict[rel]:
                single_rule = verbalize_rule(rule, self.id2relation) + "\n"
                part = re.split(r'\s+', single_rule.strip())
                rule_with_confidence = f"{part[-1]}"
                rules_var[rule_with_confidence] = rule
                rules_str += single_rule
        return rules_str, rules_var

    def generate_filename(self, dt, rule_lengths, num_walks, transition_distr, seed, suffix):
        filename = f"{dt}_r{rule_lengths}_n{num_walks}_{transition_distr}_s{seed}_{suffix}"
        return filename.replace(" ", "")

    def remove_first_three_columns(self, input_path, output_path):
        rule_id_content = []
        with open(input_path, 'r') as input_file, open(output_path, 'w', encoding="utf-8") as output_file:
            for line in input_file:
                columns = line.split()
                new_line = ' '.join(columns[3:])
                new_line_for_rule_id = ' '.join(columns[3:]) + '&' + columns[0] + '\n'
                rule_id_content.append(new_line_for_rule_id)
                output_file.write(new_line + '\n')
        return rule_id_content

    def parse_and_save_rules(self, remove_filename, keys, relation_regex, output_filename):
        output_file_path = maybe_windows_long_path(os.path.join(self.output_dir, output_filename))
        with open(maybe_windows_long_path(os.path.join(self.output_dir, remove_filename)), 'r') as file:
            lines = file.readlines()
            converted_rules = parse_rules_for_path(lines, keys, relation_regex)
        with open(output_file_path, 'w') as file:
            for head, paths in converted_rules.items():
                json.dump({"head": head, "paths": paths}, file)
                file.write('\n')
        print(f'Rules have been converted and saved to {output_file_path}')
        return converted_rules

    def parse_and_save_rules_with_names(self, remove_filename, rel2idx, relation_regex, output_filename,
                                        rule_id_content):
        input_file_path = maybe_windows_long_path(os.path.join(self.output_dir, remove_filename))
        output_file_path = maybe_windows_long_path(os.path.join(self.output_dir, output_filename))
        with open(input_file_path, 'r') as file:
            rules_content = file.readlines()
            rules_name_dict = parse_rules_for_name(rules_content, list(rel2idx.keys()), relation_regex)
        with open(output_file_path, 'w') as file:
            json.dump(rules_name_dict, file, indent=4)
        print(f'Rules have been converted and saved to {output_file_path}')

    def parse_and_save_rules_with_ids(self, rule_id_content, rel2idx, relation_regex, output_filename):
        output_file_path = maybe_windows_long_path(os.path.join(self.output_dir, output_filename))
        rules_id_dict = parse_rules_for_id(rule_id_content, rel2idx, relation_regex)
        with open(output_file_path, 'w') as file:
            json.dump(rules_id_dict, file, indent=4)
        print(f'Rules have been converted and saved to {output_file_path}')

    def save_rule_name_with_confidence(self, file_path, relation_regex, out_file_path, relations):
        rules_dict = {}
        with open(file_path, 'r') as fin:
            rules = fin.readlines()
            for rule in rules:
                # Split the string by spaces to get the columns
                columns = rule.split()

                # Extract the first and fourth columns
                first_column = columns[0]
                fourth_column = ''.join(columns[3:])
                output = f"{fourth_column}&{first_column}"

                regrex_list = fourth_column.split('<-')
                match = re.search(relation_regex, regrex_list[0])
                if match:
                    head = match[1].strip()
                    if head not in relations:
                        raise ValueError(f"Not exist relation:{head}")
                else:
                    continue

                if head not in rules_dict:
                    rules_dict[head] = []
                rules_dict[head].append(output)
        save_json_data(rules_dict, out_file_path)


def parse_rules_for_path(lines, relations, relation_regex):
    converted_rules = {}
    for line in lines:
        rule = line.strip()
        if not rule:
            continue
        temp_rule = re.sub(r'\s*<-\s*', '&', rule)
        regrex_list = temp_rule.split('&')

        head = ""
        body_list = []
        for idx, regrex_item in enumerate(regrex_list):
            match = re.search(relation_regex, regrex_item)
            if match:
                rel_name = match.group(1).strip()
                if rel_name not in relations:
                    raise ValueError(f"Not exist relation:{rel_name}")
                if idx == 0:
                    head = rel_name
                    paths = converted_rules.setdefault(head, [])
                else:
                    body_list.append(rel_name)

        path = '|'.join(body_list)
        paths.append(path)

    return converted_rules


def parse_rules_for_name(lines, relations, relation_regex):
    rules_dict = {}
    for rule in lines:
        temp_rule = re.sub(r'\s*<-\s*', '&', rule)
        regrex_list = temp_rule.split('&')
        match = re.search(relation_regex, regrex_list[0])
        if match:
            head = match[1].strip()
            if head not in relations:
                raise ValueError(f"Not exist relation:{head}")
        else:
            continue

        if head not in rules_dict:
            rules_dict[head] = []
        rules_dict[head].append(rule)

    return rules_dict


def parse_rules_for_id(rules, rel2idx, relation_regex):
    rules_dict = {}
    for rule in rules:
        temp_rule = re.sub(r'\s*<-\s*', '&', rule)
        regrex_list = temp_rule.split('&')
        match = re.search(relation_regex, regrex_list[0])
        if match:
            head = match[1].strip()
            if head not in rel2idx:
                raise ValueError(f"Relation '{head}' not found in rel2idx")
        else:
            continue

        rule_id = rule2id(rule.rsplit('&', 1)[0], rel2idx, relation_regex)
        rule_id = rule_id + '&' + rule.rsplit('&', 1)[1].strip()
        rules_dict.setdefault(head, []).append(rule_id)
    return rules_dict


def rule2id(rule, relation2id, relation_regex):
    temp_rule = copy.deepcopy(rule)
    temp_rule = re.sub(r'\s*<-\s*', '&', temp_rule)
    temp_rule = temp_rule.split('&')
    rule2id_str = ""

    try:
        for idx, _ in enumerate(temp_rule):
            match = re.search(relation_regex, temp_rule[idx])
            rel_name = match[1].strip()
            subject = match[2].strip()
            object = match[3].strip()
            timestamp = match[4].strip()
            rel_id = relation2id[rel_name]
            full_id = f"{rel_id}({subject},{object},{timestamp})"
            if idx == 0:
                full_id = f"{full_id}<-"
            else:
                full_id = f"{full_id}&"

            rule2id_str += f"{full_id}"
    except KeyError as keyerror:
        # 捕获异常并打印调用栈信息
        traceback.print_exc()
        raise ValueError(f"KeyError: {keyerror}")

    except Exception as e:
        raise ValueError(f"An error occurred: {rule}")

    return rule2id_str[:-1]


def verbalize_rule(rule, id2relation):
    """
    Verbalize the rule to be in a human-readable format.

    Parameters:
        rule (dict): rule from Rule_Learner.create_rule
        id2relation (dict): mapping of index to relation

    Returns:
        rule_str (str): human-readable rule
    """

    var_constraints = [
        list(x)
        for x in normalize_var_constraints(rule["var_constraints"], len(rule["body_rels"]))
    ]

    rule_str = "{0:8.6f}  {1:4}  {2:4}  {3}(X0,X{4},T{5})<-"
    obj_idx = [
        idx
        for idx in range(len(var_constraints))
        if len(rule["body_rels"]) in var_constraints[idx]
    ][0]
    rule_str = rule_str.format(
        rule["conf"],
        rule["rule_supp"],
        rule["body_supp"],
        id2relation[rule["head_rel"]],
        obj_idx,
        len(rule["body_rels"]),
    )

    for i in range(len(rule["body_rels"])):
        sub_idx = [
            idx for idx in range(len(var_constraints)) if i in var_constraints[idx]
        ][0]
        obj_idx = [
            idx for idx in range(len(var_constraints)) if i + 1 in var_constraints[idx]
        ][0]
        rule_str += "{0}(X{1},X{2},T{3})&".format(
            id2relation[rule["body_rels"][i]], sub_idx, obj_idx, i
        )

    return rule_str[:-1]


def verbalize_concrete_rule(rule, walk, id2relation, id2entity, id2ts):
    """
    Verbalize a rule with concrete entities and timestamps from the generating walk.
    """
    # Align entities/timestamps with the reversed body order used in rule construction.
    entities_seq = walk["entities"][1:][::-1]
    ts_seq = walk["timestamps"][1:][::-1]
    head_ts = walk["timestamps"][0]
    var_constraints = [
        list(x)
        for x in normalize_var_constraints(rule["var_constraints"], len(rule["body_rels"]))
    ]

    # Map variable index -> representative entity id
    var_entities = {}
    for var_idx, positions in enumerate(var_constraints):
        if not positions:
            continue
        representative_pos = positions[0]
        if representative_pos >= len(entities_seq):
            representative_pos = len(entities_seq) - 1
        var_entities[var_idx] = entities_seq[representative_pos]

    def find_var_idx(pos):
        for idx, group in enumerate(var_constraints):
            if pos in group:
                return idx
        raise ValueError(f"Position {pos} not found in var_constraints")

    head_obj_idx = find_var_idx(len(rule["body_rels"]))
    head_sub_ent = var_entities.get(0, entities_seq[0])
    head_obj_ent = var_entities.get(head_obj_idx, entities_seq[-1])

    rule_str = "{0}({1},{2},{3})<-".format(
        id2relation[rule["head_rel"]],
        id2entity[head_sub_ent],
        id2entity[head_obj_ent],
        id2ts[head_ts],
    )

    for i in range(len(rule["body_rels"])):
        sub_idx = find_var_idx(i)
        obj_idx = find_var_idx(i + 1)
        sub_ent = var_entities.get(sub_idx, entities_seq[sub_idx])
        obj_ent = var_entities.get(obj_idx, entities_seq[obj_idx])
        ts_val = ts_seq[i] if i < len(ts_seq) else head_ts
        rule_str += "{0}({1},{2},{3})&".format(
            id2relation[rule["body_rels"][i]],
            id2entity[sub_ent],
            id2entity[obj_ent],
            id2ts[ts_val],
        )

    return rule_str[:-1]


def rules_statistics(rules_dict):
    """
    Show statistics of the rules.

    Parameters:
        rules_dict (dict): rules

    Returns:
        None
    """

    print(
        "Number of relations with rules: ", len(rules_dict)
    )  # Including inverse relations
    print("Total number of rules: ", sum([len(v) for k, v in rules_dict.items()]))

    lengths = []
    for rel in rules_dict:
        lengths += [len(x["body_rels"]) for x in rules_dict[rel]]
    rule_lengths = [(k, v) for k, v in Counter(lengths).items()]
    print("Number of rules by length: ", sorted(rule_lengths))
