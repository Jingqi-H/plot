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
    'Training': '#4C72B0',
    'Internal': '#55A868',
    'External': '#DD8452'
}
colors = [color_map[g] for g in groups]

# ---------------------- 极坐标 ----------------------
N = len(labels)
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
width = 2 * np.pi / N * 0.8

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# 🔥 空心
inner_radius = max(sizes) * 0.8

bars = ax.bar(
    theta,
    sizes,
    width=width,
    bottom=inner_radius,
    color=colors,
    edgecolor='white',
    linewidth=1
)

# ---------------------- 去掉所有文字 ----------------------
ax.set_xticks([])
ax.set_yticklabels([])

# ---------------------- 虚线参考线（关键） ----------------------
ax.yaxis.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)
ax.xaxis.grid(False)

# ---------------------- 美化 ----------------------
ax.spines['polar'].set_visible(False)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# ---------------------- 保存 ----------------------
plt.tight_layout()
plt.savefig(f'PLOTS/0.tableone/radial_bar_{plot_case}_minimal.svg', dpi=300, bbox_inches='tight')
plt.show()
plt.close()