import pandas as pd
import matplotlib.pyplot as plt
import os

FONTSIZE = 14
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = FONTSIZE

NUM_PATH = 'PLOTS/0.tableone/各中心的样本数.csv'
plot_case = 'slide'

# ---------------------- 读取数据 ----------------------
plot_df = pd.read_csv(NUM_PATH)

labels = plot_df['cohort'].tolist()
sizes = plot_df[plot_case + '_num'].tolist()

# ---------------------- 颜色映射 ----------------------
def get_color(name):
    name_lower = name.lower()
    if 'sxch' in name_lower and 'train' in name_lower:
        return '#fff2ca'   # training
    elif 'sxch' in name_lower and ('val' in name_lower or 'internal' in name_lower):
        return '#fce6d5'   # internal validation
    else:
        return '#f4f6ed'   # external validation

colors = [get_color(n) for n in labels]

# ---------------------- 数据反转（保证从上到下） ----------------------
dataset_names_reversed = labels[::-1]
sample_counts_reversed = sizes[::-1]
colors_reversed = colors[::-1]

# ---------------------- 绘图 ----------------------
fig, ax = plt.subplots(figsize=(7, 4))

bars = ax.barh(
    range(len(dataset_names_reversed)),
    sample_counts_reversed,
    color=colors_reversed,
    height=0.7,   # ⭐ 更紧凑
    edgecolor='white',
    linewidth=1
)

# 对数坐标
ax.set_xscale('log')

# ---------------------- 数值标注 ----------------------
for bar, count in zip(bars, sample_counts_reversed):
    width = bar.get_width()
    ax.text(
        width * 1.08,   # ⭐ 稍微远一点更清晰
        bar.get_y() + bar.get_height() / 2,
        f'{count:,}',
        ha='left',
        va='center',
        fontsize=11,
        fontweight='bold'
    )

# ---------------------- 坐标轴 ----------------------
ax.set_xlabel('')
ax.set_ylabel('')

ax.set_yticks(range(len(dataset_names_reversed)))
ax.set_yticklabels(dataset_names_reversed)

# X轴到顶部
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# 去边框
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)

# X轴格式
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{int(x):,}')
)

# ---------------------- Legend ----------------------
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#fff2ca', label='Training'),
    Patch(facecolor='#fce6d5', label='Internal Validation'),
    Patch(facecolor='#f4f6ed', label='External Validation')
]
ax.legend(handles=legend_elements, frameon=False, loc='lower right', fontsize=FONTSIZE)

# ---------------------- 进一步压缩间距 ----------------------
ax.margins(y=0.02)  # ⭐ 减少上下空白


# 设置title,放在画布上方
ax.set_title(f'Number of Slides: {sum(sample_counts_reversed):,}', fontsize=FONTSIZE,
    pad=20,  # 标题和图表之间的距离（关键参数）
    loc='center'  # 居中显示
    )

plt.tight_layout()
plt.savefig(f'PLOTS/0.tableone/bar_{plot_case}.svg', dpi=300, bbox_inches='tight')
plt.savefig(f'PLOTS/0.tableone/bar_{plot_case}.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()