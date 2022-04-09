import jsonlines
import os
import json

lama_rel_patterns = {}

with jsonlines.open('../LAMA/raw_data/relations.jsonl', 'r') as r_reader:
    for pattern in r_reader:
        lama_rel_patterns[pattern['relation']] = pattern

data_11 = []
data_n1 = []
data_nm = []
data_all = []

TREx_dir = '../LAMA/raw_data/TREx/'
pararel_pattern_dir = 'raw_data/patterns'
for pattern_filename in os.listdir(pararel_pattern_dir):
    pattern_pathname = os.path.join(pararel_pattern_dir, pattern_filename)
    relation = pattern_filename.split('.')[0]
    TREx_pathname = os.path.join(TREx_dir, pattern_filename)
    if not os.path.exists(TREx_pathname):
        continue
    pararel_rel_patterns = []
    with jsonlines.open(pattern_pathname, 'r') as p_reader:
        for pattern in p_reader:
            pararel_rel_patterns.append(pattern)
    # filter out relations with <= 3 templates
    if len(pararel_rel_patterns) <= 3:
        continue
    with jsonlines.open(TREx_pathname, 'r') as t_reader:
        rel_pattern = lama_rel_patterns[relation]
        for entry in t_reader:
            subj = entry['sub_label']
            obj = entry['obj_label']
            entry_bag = []
            for pattern in pararel_rel_patterns:
                prompt = pattern['pattern'].replace('[X]', subj).replace('[Y]', '[MASK]')
                answer = obj
                rel_label = relation + '(' + rel_pattern['label'] + ')'
                entry = [prompt, answer, rel_label]
                entry_bag.append(entry)  # multiple entries per bag
            if rel_pattern['type'] == '1-1':
                data_11.append(entry_bag)
            if rel_pattern['type'] == 'N-1':
                data_n1.append(entry_bag)
            if rel_pattern['type'] == 'N-M':
                data_nm.append(entry_bag)
            data_all.append(entry_bag)

with open('data_11.json', 'w') as fw:
    json.dump(data_11, fw, ensure_ascii=False, indent=2)
with open('data_n1.json', 'w') as fw:
    json.dump(data_n1, fw, ensure_ascii=False, indent=2)
with open('data_nm.json', 'w') as fw:
    json.dump(data_nm, fw, ensure_ascii=False, indent=2)
with open('data_all.json', 'w') as fw:
    json.dump(data_all, fw, ensure_ascii=False, indent=2)