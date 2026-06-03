import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

FONTSIZE = 15
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = FONTSIZE

NUM_PATH = 'PLOTS/0.tableone/各中心的样本数.csv'
plot_case = 'slide'

# ---------------------- 读取数据 ----------------------
plot_df = pd.read_csv(NUM_PATH)

labels = plot_df['cohort'].tolist()
sizes = plot_df[plot_case + '_num'].tolist()

# ---------------------- log转换（核心） ----------------------
log_sizes = np.log10(sizes)

# ---------------------- 颜色映射 ----------------------
def get_color(name):
    name_lower = name.lower()
    if 'sxch' in name_lower and 'train' in name_lower:
        return '#ee822f'
    elif 'sxch' in name_lower and ('val' in name_lower or 'internal' in name_lower):
        return '#f2ba02'
    else:
        return '#75bd42'

colors = [get_color(n) for n in labels]

# ---------------------- 控制间距 ----------------------
x = np.arange(len(labels)) * 0.85
bar_width = 0.6

# ---------------------- 绘图 ----------------------
fig, ax = plt.subplots(figsize=(5,7))

bars = ax.bar(
    x,
    log_sizes,   # ⭐ 用log后的值
    width=bar_width,
    color=colors,
    edgecolor='white',
    linewidth=1
)

# ---------------------- 标注原始数值 ----------------------
for bar, raw_value in zip(bars, sizes):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.05,
        f'{raw_value:,}',   # ⭐ 原始值
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold'
    )

# ---------------------- Y轴显示为原始刻度 ----------------------
# 手动设置tick（关键！）
yticks = np.arange(int(min(log_sizes)), int(max(log_sizes)) + 1)
ax.set_yticks(yticks)
ax.set_yticklabels([f'{int(10**y):,}' for y in yticks])
ax.set_ylim(np.log10(90), max(log_sizes) + 0.2)

ax.set_ylabel('Number of WSIs (log scale)')
ax.set_xlabel(f'Total: {sum(sizes):,} WSIs',) # 放在顶部

# X轴
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')

# 去边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------------------- Legend ----------------------
legend_elements = [
    Patch(facecolor='#ee822f', label='Training'),
    Patch(facecolor='#f2ba02', label='Internal Validation'),
    Patch(facecolor='#75bd42', label='External Validation')
]
ax.legend(handles=legend_elements, frameon=False, loc='upper right', fontsize=12)

# ---------------------- 间距优化 ----------------------
ax.margins(x=0.01)

plt.tight_layout()
plt.savefig(f'PLOTS/0.tableone/bar_log_visual_{plot_case}.svg', dpi=300, bbox_inches='tight')
plt.show()
plt.close()