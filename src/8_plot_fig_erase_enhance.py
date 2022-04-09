import json
from matplotlib import pyplot as plt
import numpy as np
import os

kn_dir = '../results/kn/'
fig_dir = '../results/figs/'
with open(os.path.join(kn_dir, 'modify_activation_rlt.json'), 'r') as f:
    modified_rlts = json.load(f)
with open(os.path.join(kn_dir, 'base_modify_activation_rlt.json'), 'r') as f:
    base_modified_rlts = json.load(f)

rel_set = set()

for k, v in modified_rlts.items():
    rel = k.split('-')[-1]
    rel_set.update([rel])

rel_list = sorted(list(rel_set))

# ================================== suppress ===========================================
plt.figure(figsize=(22, 5.5), dpi=100)

x_labels = []
y_values = []
for rel in rel_list:
    x_labels.append(rel)
    key = 'kn_bag-' + rel
    base_key = 'base_kn_bag-' + rel
    tmp_ys = []
    tmp_ys.append(modified_rlts[key]['rm_own:ave_delta_ratio'])
    tmp_ys.append(base_modified_rlts[base_key]['rm_own:ave_delta_ratio'])
    y_values.append(tmp_ys)

plt.ylabel('Correct Probability Change Ratio', fontsize=18)

x1 = np.array([1 + i * 3 for i in range(len(rel_list))])
x2 = x1 + 1
ys = np.array(y_values)
print(ys.mean(axis=0))

plt.xticks([i * 3 + 1.5 for i in range(len(x_labels))], labels=x_labels)
plt.bar(x1, ys[:, 0], width=1, edgecolor='black', hatch="//", color='#0165fc', label="Ours")
plt.bar(x2, ys[:, 1], width=1, edgecolor='black', hatch="\\\\", color='#bfefff', label="Baseline")
plt.yticks(np.arange(-0.6, 0.3, 0.1), [f'{y}%' for y in np.arange(-60, 30, 10)], fontsize=20)
plt.xlim(-1, 3 * len(x_labels) + 1)
plt.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True, rotation=25, labelsize=18)
plt.grid(True, axis='y', alpha=0.3)

plt.legend(loc='upper right', fontsize=20)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'suppress.pdf'))
plt.close()

# ================================== amplify ===========================================

plt.figure(figsize=(22, 5.5), dpi=100)

x_labels = []
y_values = []
for rel in rel_list:
    x_labels.append(rel)
    key = 'kn_bag-' + rel
    base_key = 'base_kn_bag-' + rel
    tmp_ys = []
    tmp_ys.append(modified_rlts[key]['eh_own:ave_delta_ratio'])
    tmp_ys.append(base_modified_rlts[base_key]['eh_own:ave_delta_ratio'])
    y_values.append(tmp_ys)

plt.ylabel('Correct Probability Change Ratio', fontsize=18)

x1 = np.array([1 + i * 3 for i in range(len(rel_list))])
x2 = x1 + 1
ys = np.array(y_values)
print(ys.mean(axis=0))

plt.xticks([i * 3 + 1.5 for i in range(len(x_labels))], labels=x_labels)
plt.bar(x1, ys[:, 0], width=1, edgecolor='black', hatch="//", color='#e50000', label="Ours")
plt.bar(x2, ys[:, 1], width=1, edgecolor='black', hatch="\\\\", color='#ffe4e1', label="Baseline")
plt.yticks(np.arange(-0.1, 0.9, 0.1), [f'{y}%' for y in np.arange(-10, 90, 10)], fontsize=20)
plt.xlim(-1, 3 * len(x_labels) + 1)
plt.tick_params(axis="x", rotation=25, labelsize=18)
plt.grid(True, axis='y', alpha=0.3)

plt.legend(loc='upper right', fontsize=20, bbox_to_anchor=(0.96, 0.99))

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'amplify.pdf'))
plt.close()