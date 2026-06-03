import pandas as pd
import matplotlib.pyplot as plt
import os

FONTSIZE = 15
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = FONTSIZE

NUM_PATH = 'PLOTS/0.tableone/各中心的样本数.csv'
plot_case = 'case'

# ---------------------- 3. 读取CSV并绘制饼图 ----------------------
# 读取CSV文件（验证数据，也可直接用上面的csv_df）
plot_df = pd.read_csv(NUM_PATH)



# 准备绘图数据
labels = plot_df['cohort'].tolist()
sizes = plot_df[plot_case+'_num'].tolist()
colors = ['#D8E6E7'] * len(labels)  # 配色（避免单调）
# colors = ['#D4DFE6', '#d6ecfa', '#EDAFB8', '#96CEB4', '#f7e1d7', '#dedbd2', '#b0c4b1']  # 配色（避免单调）


# ---------------------- 数据准备 ----------------------
# 数据集名称（替换为你的实际名称，如"SXCH"、"YYH"等）
dataset_names = labels.copy()
sample_counts = sizes.copy()
print(sample_counts)
# ---------------------- 绘图设置 ----------------------

fig, ax = plt.subplots(figsize=(10, 5))

# 反转数据集顺序，使第一个标签在顶部（从上到下显示）
dataset_names_reversed = dataset_names[::-1]
sample_counts_reversed = sample_counts[::-1]
colors_reversed = colors[::-1]

# 绘制横向条形图（X轴对数刻度）
bars = ax.barh(
    range(len(dataset_names_reversed)), sample_counts_reversed, 
    color=colors_reversed, height=0.6, edgecolor='white', linewidth=1
)

# 设置X轴为对数刻度（核心！适配量级差异）
ax.set_xscale('log')

# ---------------------- 美化与标注（关键：显示具体数值） ----------------------
# 给每个条形标注实际数值（避免对数刻度导致数值不直观）
for i, (bar, count) in enumerate(zip(bars, sample_counts_reversed)):
    width = bar.get_width()
    ax.text(
        width + width * 0.05, bar.get_y() + bar.get_height()/2,  # 数值标注位置
        f'{count:,}',  # 千分位分隔符（如18000→18,000）
        ha='left', va='center', fontsize=11, fontweight='bold'
    )

# 坐标轴与标题设置
# sum(sample_counts_reversed)3位数有个逗号
ax.set_xlabel(f'{sum(sample_counts_reversed):,} Patients', fontsize=FONTSIZE)
ax.set_ylabel('')  # 清空Y轴标签
# ax.set_title('Sample Size of Datasets (Log Scale)', fontsize=14, fontweight='bold', pad=20)

# 设置Y轴刻度和标签（从上到下显示labels顺序）
ax.set_yticks(range(len(dataset_names_reversed)))
ax.set_yticklabels(dataset_names_reversed)

# 将X轴移到顶部
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# 隐藏不需要的边框
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整X轴刻度标签（可选：显示原始数值，而非对数）
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

# 确保Y轴标签完全显示
plt.yticks(rotation=0)

# 保存高清图片
plt.tight_layout()
plt.savefig(f'PLOTS/0.tableone/bar_{plot_case}.svg', dpi=300, bbox_inches='tight')
# plt.show()