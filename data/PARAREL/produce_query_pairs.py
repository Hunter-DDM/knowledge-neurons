import jsonlines
import os
import json

lama_rel_patterns = {}

with jsonlines.open('../LAMA/raw_data/relations.jsonl', 'r') as r_reader:
    for pattern in r_reader:
        lama_rel_patterns[pattern['relation']] = pattern

query_pairs_per_rel = {}

TREx_dir = '../LAMA/raw_data/TREx/'
pararel_pattern_dir = 'raw_data/patterns'
for pattern_filename in os.listdir(pararel_pattern_dir):
    relation = pattern_filename.split('.')[0]
    TREx_pathname = os.path.join(TREx_dir, pattern_filename)
    if not os.path.exists(TREx_pathname):
        continue
    pattern_pathname = os.path.join(pararel_pattern_dir, pattern_filename)
    pararel_rel_patterns = []
    with jsonlines.open(pattern_pathname, 'r') as p_reader:
        for pattern in p_reader:
            pararel_rel_patterns.append(pattern)
    # filter out relations with <= 3 templates
    if len(pararel_rel_patterns) <= 3:
        continue
    query_pairs_per_rel[relation] = []
    with jsonlines.open(TREx_pathname, 'r') as t_reader:
        for entry in t_reader:
            subj = entry['sub_label']
            obj = entry['obj_label']
            query_pairs_per_rel[relation].append([subj, obj])
            # if len(query_pairs_per_rel[relation]) == 50:
            #     break

with open('query_pairs.json', 'w') as fw:
    json.dump(query_pairs_per_rel, fw, ensure_ascii=False, indent=2)