import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

fontsize = 14

# exp_id = '0.0001loadloss'
exp_id = 'final'
result_dir = 'PLOTS/5.热图/' + exp_id
# cmap = sns.diverging_palette(200, 20, sep=16, as_cmap=True)
cmap = 'Blues'

# 2 加载all_gates.npz
data = np.load(f'{result_dir}/all_gates.npz')
# 查看所有的slide_id
slide_ids = data.files
print("共有",len(slide_ids),"个slide")
# # data是加载进来的
num_slide = len(data)
all_data = np.zeros((len(data), 5,8))
for i, slide_id in enumerate(data.keys()):
    all_data[i] = data[slide_id]
print(all_data.shape)

# import pandas as pd
# df = pd.read_csv(f'PLOTS/5.热图/{exp_id}_SXCH-Val-用于筛选样本.csv', dtype={'slide_id_wax':str, 'case_id':str})
# # 2 加载all_gates.npz
# data = np.load(f'{result_dir}/all_gates.npz')
# print(len(df['slide_id_wax'].values.tolist()))
# all_data = np.zeros((len(df['slide_id_wax'].values.tolist()), 5,8))
# ids = []
# num = 0
# for i, slide_id in enumerate(data.keys()):
#         if slide_id in df['slide_id_wax'].values.tolist():
#             all_data[num] = data[slide_id]
#             ids.append(slide_id)
#             num += 1
#         else:
#             continue
# slide_ids = ids
# print(len(slide_ids))

# count_matrix = np.mean(all_data, axis=0)
# print(count_matrix)
# expert_num = 8
# task_num = 5
# fontsize = 12
# task_type =['Survival','Grade', 'Histo Type', 'Lauren Type', 'Stage']
# # 绘制热力图
# plt.figure(figsize=(10, 5))
# ax = sns.heatmap(count_matrix, annot=True, fmt=".2f", cmap=cmap, # Blues
#             xticklabels=[f"{i+1}" for i in range(expert_num)],
#             yticklabels=[f"{task_type[i]}" for i in range(task_num)],
#             square=True,
#             cbar=True,cbar_kws={"shrink": 1},
#             annot_kws={"size": fontsize-5}
#             )
# # 调整colorbar 高度与热力图一致
# cbar = ax.collections[0].colorbar
# # cbar.ax.set_aspect('auto')
# # cbar.ax.set_box_aspect(1)
# cbar.ax.tick_params(labelsize=fontsize-5) # 设置字体大小

# plt.xlabel("Experts",fontsize=fontsize)
# plt.ylabel("Tasks",fontsize=fontsize)
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)

# plt.tight_layout()
# plt.savefig(f'{result_dir}/专家权重-热图.svg', bbox_inches='tight', pad_inches=0)

# plt.show()
# plt.close()


# 初始化统计矩阵 (任务数=4, 专家数=8)
task_num, expert_num = 5, 8
count_matrix = np.zeros((task_num, expert_num), dtype=int)
task_type =['Survival','Grade', 'Histo.', 'Lauren', 'Stage']

# 遍历所有 slide
for slide_id in data.files:
    arr = data[slide_id]  # shape (4,8)
    for task_idx in range(task_num):
        # 取出当前任务的贡献度
        contrib = arr[task_idx]

        # 选出贡献度最大的 4 个专家
        top_experts = contrib.argsort()[-4:]

        # 统计
        for exp in top_experts:
            count_matrix[task_idx, exp] += 1


# 计算百分比
percent_matrix = np.zeros_like(count_matrix, dtype=float)
for task_idx in range(task_num):
    total = len(slide_ids) # 总样本数
    if total > 0:
        print(count_matrix[task_idx])
        percent_matrix[task_idx] = np.round(count_matrix[task_idx] / total, 2)

print(percent_matrix)
# 绘制热力图
plt.figure(figsize=(10, 6))
# plt.figure()
# ax = sns.heatmap(count_matrix, annot=True, fmt="d", cmap="Blues",
#             xticklabels=[f"{i+1}" for i in range(expert_num)],
#             yticklabels=[f"{task_type[i]}" for i in range(task_num)],
#             square=True,
#             cbar=True,cbar_kws={"shrink": 1},
#             annot_kws={"size": fontsize-5}
#             )
# 不显示文字
ax = sns.heatmap(percent_matrix, fmt=".2f", cmap=cmap,
            xticklabels=[f"{i+1}" for i in range(expert_num)],
            yticklabels=[f"{task_type[i]}" for i in range(task_num)],
            # square=True,
            cbar=True,cbar_kws={"shrink": 1},
            annot_kws={"size": fontsize},
            annot=True,
            # annot=False,
            linewidths=1.2,    # 稍粗的边界线
            linecolor="#FFFFFF"# 白色（十六进制写法，和"white"等效）
            )

# 调整colorbar 高度与热力图一致
cbar = ax.collections[0].colorbar
# cbar.ax.set_aspect('auto')
# cbar.ax.set_box_aspect(1)
cbar.ax.tick_params(labelsize=fontsize) # 设置字体大小
# 设置bar的字只显示0和1
cbar.set_ticks([0, 1])

plt.xlabel("Experts",fontsize=fontsize)
# plt.ylabel("Tasks",fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize, rotation=90)

plt.tight_layout()
plt.savefig(f'{result_dir}/专家使用频次-热图-百分比.svg', bbox_inches='tight', pad_inches=0)

plt.show()
plt.close()

