import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

import numpy as np
import plotly.graph_objects as go

np.random.seed(42)


exp_id = '0.01loadloss'
result_dir = 'PLOTS/5.热图/' + exp_id



df = pd.read_csv(f'PLOTS/5.热图/{exp_id}_SXCH-Val-用于筛选样本.csv', dtype={'slide_id_wax':str, 'case_id':str})


# 2 加载all_gates.npz
data = np.load(f'{result_dir}/all_gates.npz')
print(len(df['slide_id_wax'].values.tolist()))
all_data = np.zeros((len(df['slide_id_wax'].values.tolist()), 5,8))
ids = []
num = 0
for i, slide_id in enumerate(data.keys()):
        if slide_id in df['slide_id_wax'].values.tolist():
            all_data[num] = data[slide_id]
            ids.append(slide_id)
            num += 1
        else:
            continue
# print(all_data.shape)
# print(len(ids))

# 筛选ids, 只保留df中slide_id_wax在ids中的样本
missing = set(df['slide_id_wax'].values.tolist()) - set(ids)
print('missing:', missing)

df_new = (
    df.set_index('slide_id_wax')
      .loc[ids]
      .reset_index()[['slide_id_wax', 'risk_group']]
)
risk_group = df_new['risk_group'].values.astype(int)


weights = all_data.copy()
tasks = ["T1 survival","T2 grade","T3 type","T4 lauren","T5 stage"]
experts = [f"Expert {i+1}" for i in range(8)]
risk_labels = ["Low risk","High risk"]

# -----------------------
# Step1: 取每个任务 top4专家
# -----------------------
topk = 4
topk_idx = np.argsort(weights, axis=-1)[:, :, -topk:]  # shape (3707,5,4)

# -----------------------
# Step2: 统计 Task -> Expert 的频次
# -----------------------
task_expert_counts = np.zeros((len(tasks), len(experts)), dtype=int)

for i in range(weights.shape[0]):
    for t in range(len(tasks)):
        for e in topk_idx[i,t]:
            task_expert_counts[t,e] += 1
print('task_expert_counts:\n',task_expert_counts)

# -----------------------
# Step3: 统计 Expert -> Risk 的频次
# -----------------------
expert_risk_counts = np.zeros((len(experts), len(risk_labels)), dtype=int)

for i in range(weights.shape[0]):
    for t in range(len(tasks)):
        for e in topk_idx[i,t]:
            expert_risk_counts[e, risk_group[i]] += 1
print('expert_risk_counts:\n',expert_risk_counts)

# -----------------------
# Step4: 构建桑基图节点和边
# -----------------------
node_labels = tasks + experts + risk_labels
node_colors = ["#636EFA"]*len(tasks) + ["#EF553B"]*len(experts) + ["#00CC96"]*len(risk_labels)

# 边 Task -> Expert
source = []
target = []
value = []

for t_idx, task in enumerate(tasks):
    for e_idx, expert in enumerate(experts):
        count = task_expert_counts[t_idx,e_idx]
        if count > 0:
            source.append(t_idx)                    # Task节点索引
            target.append(len(tasks)+e_idx)        # Expert节点索引
            value.append(count)

# 边 Expert -> Risk
for e_idx, expert in enumerate(experts):
    for r_idx, risk in enumerate(risk_labels):
        count = expert_risk_counts[e_idx,r_idx]
        if count > 0:
            source.append(len(tasks)+e_idx)        # Expert节点索引
            target.append(len(tasks)+len(experts)+r_idx)  # Risk节点索引
            value.append(count)

# -----------------------
# Step5: 画桑基图
# -----------------------
fig = go.Figure(data=[go.Sankey(
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = node_labels,
        color = node_colors
    ),
    link = dict(
        source = source,
        target = target,
        value = value
    ))])

fig.update_layout(title_text="Task -> Expert -> Risk Sankey Diagram (Top4 Experts)", font_size=12)
fig.show()