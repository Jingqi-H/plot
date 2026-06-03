import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 15
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = FONTSIZE

NUM_PATH = 'PLOTS/0.tableone/各中心的样本数.csv'
plot_case = 'slide'

# ---------------------- 读取数据 ----------------------
df = pd.read_csv(NUM_PATH)

labels = df['cohort'].tolist()
sizes = df[plot_case + '_num'].tolist()
# log 2
sizes = np.log2(sizes)

# ---------------------- 分类 ----------------------
groups = []
for name in labels:
    if name == 'SXCH Training':
        groups.append('Training')
    elif name == 'SXCH Validation':
        groups.append('Internal')
    else:
        groups.append('External')

# ---------------------- 排序 ----------------------
data = list(zip(labels, sizes, groups))
data.sort(key=lambda x: x[1], reverse=True)
labels, sizes, groups = zip(*data)

# ---------------------- 颜色 ----------------------
color_map = {
    'Training': '#ee822f',
    'Internal': '#f2ba02',
    'External': '#75bd42'
}
colors = [color_map[g] for g in groups]

# ---------------------- 极坐标 ----------------------
N = len(labels)
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
width = 2 * np.pi / N * 0.98

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))

# 🔥 核心：设置空心半径
inner_radius = max(sizes) * 0.8   # 可以调（0.2~0.4都可以）

bars = ax.bar(
    theta,
    sizes,
    width=width,
    bottom=inner_radius,   # 👈 关键
    color=colors,
    edgecolor='white',
    linewidth=1
)

# ---------------------- 数值标注 ----------------------
max_val = max(sizes)

for angle, radius in zip(theta, sizes):
    ax.text(
        angle,
        inner_radius + radius + max_val * 0.05,
        f'{radius:,}',
        ha='center',
        va='center',
        fontsize=10
    )

# ---------------------- 标签旋转 ----------------------
ax.set_xticks(theta)
ax.set_xticklabels(labels)

for label, angle in zip(ax.get_xticklabels(), theta):
    angle_deg = np.degrees(angle)
    if angle_deg > 180:
        label.set_rotation(angle_deg + 180)
    else:
        label.set_rotation(angle_deg)
    label.set_horizontalalignment('center')

# ---------------------- 中心空白文字 ----------------------
total = sum(sizes)
ax.text(
    0, 0,
    f'Total\n{total:,}',
    ha='center',
    va='center',
    fontsize=16,
    fontweight='bold'
)

# ---------------------- 美化 ----------------------
ax.set_yticklabels([])
ax.spines['polar'].set_visible(False)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# ---------------------- 图例 ----------------------
handles = [
    plt.Line2D([0], [0], color=color_map[k], lw=6)
    for k in ['Training', 'Internal', 'External']
]

ax.legend(
    handles,
    ['Training', 'Internal validation', 'External validation'],
    loc='upper right',
    bbox_to_anchor=(1.2, 1.1),
    frameon=False
)

# ---------------------- 保存 ----------------------
plt.tight_layout()
plt.savefig(f'PLOTS/0.tableone/radial_bar_{plot_case}_donut.svg', dpi=300, bbox_inches='tight')
plt.show()
plt.close()