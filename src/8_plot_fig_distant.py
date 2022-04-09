import json
from matplotlib import pyplot as plt
import numpy as np
import os

fig_dir = '../results/figs/'
distant_dir = '../results/distant/'
with open(os.path.join(distant_dir, 'kn_weights@distant_data.json'), 'r') as f:
    kn_weights = json.load(f)
with open(os.path.join(distant_dir, 'base_kn_weights@distant_data.json'), 'r') as f:
    base_kn_weights = json.load(f)

rel_set = set()

for rel in kn_weights.keys():
    rel_set.update([rel])

rel_list = sorted(list(rel_set))

rel_ave_weights_th = {}
rel_ave_weights_h = {}
rel_ave_weights_rd = {}
for rel, rel_weights in kn_weights.items():
    rel_ave_weights_th[rel] = []
    rel_ave_weights_h[rel] = []
    rel_ave_weights_rd[rel] = []
    for weights in rel_weights:
        if type(weights) == str:
            continue
        rel_ave_weights_th[rel].extend(weights[0])
        rel_ave_weights_h[rel].extend(weights[1])
        rel_ave_weights_rd[rel].extend(weights[2])
    rel_ave_weights_th[rel] = np.mean(rel_ave_weights_th[rel]) if len(rel_ave_weights_th[rel]) > 0 else 0
    rel_ave_weights_h[rel] = np.mean(rel_ave_weights_h[rel]) if len(rel_ave_weights_h[rel]) > 0 else 0
    rel_ave_weights_rd[rel] = np.mean(rel_ave_weights_rd[rel]) if len(rel_ave_weights_rd[rel]) > 0 else 0

base_rel_ave_weights_th = {}
base_rel_ave_weights_h = {}
base_rel_ave_weights_rd = {}
for rel, rel_weights in base_kn_weights.items():
    base_rel_ave_weights_th[rel] = []
    base_rel_ave_weights_h[rel] = []
    base_rel_ave_weights_rd[rel] = []
    for weights in rel_weights:
        if type(weights) == str:
            continue
        base_rel_ave_weights_th[rel].extend(weights[0])
        base_rel_ave_weights_h[rel].extend(weights[1])
        base_rel_ave_weights_rd[rel].extend(weights[2])
    base_rel_ave_weights_th[rel] = np.mean(base_rel_ave_weights_th[rel]) if len(base_rel_ave_weights_th[rel]) > 0 else 0
    base_rel_ave_weights_h[rel] = np.mean(base_rel_ave_weights_h[rel]) if len(base_rel_ave_weights_h[rel]) > 0 else 0
    base_rel_ave_weights_rd[rel] = np.mean(base_rel_ave_weights_rd[rel]) if len(base_rel_ave_weights_rd[rel]) > 0 else 0

# ============================ average weights ==============================

plt.figure(figsize=(20, 6), dpi=100)

x_labels = []
y_values = []
base_y_values = []
for rel in rel_list:
    x_labels.append(rel)
    y_values.append([rel_ave_weights_th[rel], rel_ave_weights_h[rel], rel_ave_weights_rd[rel]])
    base_y_values.append([base_rel_ave_weights_th[rel], base_rel_ave_weights_h[rel], base_rel_ave_weights_rd[rel]])

plt.ylabel('Average Activation', fontsize=20)

x1 = np.array([1 + i * 3.5 for i in range(len(rel_list))])
x2 = x1 + 1
x3 = x1 + 2
y123 = np.array(y_values)
print(np.array(y_values).mean(axis=0))
print(np.array(base_y_values).mean(axis=0))

plt.bar(x1, y123[:, 0], width=1, edgecolor='black', hatch="//", color='#e50000', label="prompts containing head and tail entities")
plt.bar(x2, y123[:, 1], width=1, edgecolor='black', hatch="\\\\", color='#fea993', label="prompts containing only head entity", tick_label=x_labels)
plt.bar(x3, y123[:, 2], width=1, edgecolor='black', hatch="x", color='#445ee4', label="randomly selected prompts")

plt.tick_params(axis="x", labelsize=11)
plt.grid(True, axis='y', alpha=0.2)

plt.legend(loc='best', fontsize=20)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'distant.pdf'))
plt.close()