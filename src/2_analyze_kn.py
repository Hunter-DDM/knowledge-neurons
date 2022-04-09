import json
from matplotlib import pyplot as plt
import numpy as np
import os
from collections import Counter
import random
import seaborn as sns
import pandas as pd
from pandas.core.frame import DataFrame

kn_dir = '../results/kn/'
fig_dir = '../results/figs/'

# =========== stat kn_bag ig ==============

y_points = []
tot_bag_num = 0
tot_rel_num = 0
tot_kneurons = 0
kn_bag_counter = Counter()
for filename in os.listdir(kn_dir):
    if not filename.startswith('kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f)
        for kn_bag in kn_bag_list:
            for kn in kn_bag:
                kn_bag_counter.update([kn[0]])
                y_points.append(kn[0])
        tot_bag_num += len(kn_bag_list)
for k, v in kn_bag_counter.items():
    tot_kneurons += kn_bag_counter[k]
for k, v in kn_bag_counter.items():
    kn_bag_counter[k] /= tot_kneurons

# average # Kneurons
print('average ig_kn', tot_kneurons / tot_bag_num)

# =========== stat kn_bag base ==============

tot_bag_num = 0
tot_rel_num = 0
tot_kneurons = 0
base_kn_bag_counter = Counter()
for filename in os.listdir(kn_dir):
    if not filename.startswith('base_kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f)
        for kn_bag in kn_bag_list:
            for kn in kn_bag:
                base_kn_bag_counter.update([kn[0]])
        tot_bag_num += len(kn_bag_list)
for k, v in base_kn_bag_counter.items():
    tot_kneurons += base_kn_bag_counter[k]
for k, v in base_kn_bag_counter.items():
    base_kn_bag_counter[k] /= tot_kneurons
# average # Kneurons
print('average base_kn', tot_kneurons / tot_bag_num)

# =========== plot knowledge neuron distribution ===========

plt.figure(figsize=(8, 3))

x = np.array([i + 1 for i in range(12)])
y = np.array([kn_bag_counter[i] for i in range(12)])
plt.xlabel('Layer', fontsize=20)
plt.ylabel('Percentage', fontsize=20)
plt.xticks([i + 1 for i in range(12)], labels=[i + 1 for i in range(12)], fontsize=20)
plt.yticks(np.arange(-0.4, 0.5, 0.1), labels=[f'{np.abs(i)}%' for i in range(-40, 50, 10)], fontsize=18)
plt.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True, rotation=0, labelsize=18)
plt.ylim(-y.max() - 0.03, y.max() + 0.03)
plt.xlim(0.3, 12.7)
bottom = -y
y = y * 2
plt.bar(x, y, width=1.02, color='#0165fc', bottom=bottom)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(fig_dir, 'kneurons_distribution.pdf'), dpi=100)
plt.close()

# ========================================================================================
#                       knowledge neuron intersection analysis
# ========================================================================================


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]


def cal_intersec(kn_bag_1, kn_bag_2):
    kn_bag_1 = set(['@'.join(map(str, kn)) for kn in kn_bag_1])
    kn_bag_2 = set(['@'.join(map(str, kn)) for kn in kn_bag_2])
    return len(kn_bag_1.intersection(kn_bag_2))

# ====== load ig kn =======

kn_bag_list_per_rel = {}
for filename in os.listdir(kn_dir):
    if not filename.startswith('kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f)
    rel = filename.split('.')[0].split('-')[1]
    kn_bag_list_per_rel[rel] = kn_bag_list

# ig inner
inner_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list)
    for i in range(0, len_kn_bag_list):
        for j in range(i + 1, len_kn_bag_list):
            kn_bag_1 = kn_bag_list[i]
            kn_bag_2 = kn_bag_list[j]
            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
            inner_ave_intersec.append(num_intersec)
inner_ave_intersec = np.array(inner_ave_intersec).mean()
print(f'ig kn has on average {inner_ave_intersec} inner kn interseciton')

# ig inter
inter_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list)
    for i in range(0, len_kn_bag_list):
        for j in range(0, 100):
            kn_bag_1 = kn_bag_list[i]
            other_rel = random.choice([x for x in kn_bag_list_per_rel.keys() if x != rel])
            other_idx = random.randint(0, len(kn_bag_list_per_rel[other_rel]) - 1)
            kn_bag_2 = kn_bag_list_per_rel[other_rel][other_idx]
            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
            inter_ave_intersec.append(num_intersec)
inter_ave_intersec = np.array(inter_ave_intersec).mean()
print(f'ig kn has on average {inter_ave_intersec} inter kn interseciton')

# ====== load base kn =======
kn_bag_list_per_rel = {}
for filename in os.listdir(kn_dir):
    if not filename.startswith('base_kn_bag-'):
        continue
    with open(os.path.join(kn_dir, filename), 'r') as f:
        kn_bag_list = json.load(f)
    rel = filename.split('.')[0].split('-')[1]
    kn_bag_list_per_rel[rel] = kn_bag_list

# base inner
inner_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list)
    for i in range(0, len_kn_bag_list):
        for j in range(i + 1, len_kn_bag_list):
            kn_bag_1 = kn_bag_list[i]
            kn_bag_2 = kn_bag_list[j]
            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
            inner_ave_intersec.append(num_intersec)
inner_ave_intersec = np.array(inner_ave_intersec).mean()
print(f'base kn has on average {inner_ave_intersec} inner kn interseciton')

# base inter
inter_ave_intersec = []
for rel, kn_bag_list in kn_bag_list_per_rel.items():
    print(f'calculating {rel}')
    len_kn_bag_list = len(kn_bag_list)
    for i in range(0, len_kn_bag_list):
        for j in range(0, 100):
            kn_bag_1 = kn_bag_list[i]
            other_rel = random.choice([x for x in kn_bag_list_per_rel.keys() if x != rel])
            other_idx = random.randint(0, len(kn_bag_list_per_rel[other_rel]) - 1)
            kn_bag_2 = kn_bag_list_per_rel[other_rel][other_idx]
            num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
            inter_ave_intersec.append(num_intersec)
inter_ave_intersec = np.array(inter_ave_intersec).mean()
print(f'base kn has on average {inter_ave_intersec} inter kn interseciton')
