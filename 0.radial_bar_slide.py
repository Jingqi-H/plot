import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 15
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = FONTSIZE

NUM_PATH = 'PLOTS/0.tableone/各中心的样本数.csv'
plot_case = 'slide'

# ---------------------- 读取数据 ----------------------
plot_df = pd.read_csv(NUM_PATH)

labels = plot_df['cohort'].tolist()
sizes = plot_df[plot_case + '_num'].tolist()

# 反转（保持你原来的顺序逻辑）
labels = labels[::-1]
sizes = sizes[::-1]
sizes = np.log10(sizes)

N = len(labels)

# ---------------------- 构造角度 ----------------------
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

# 每个bar宽度
width = 2 * np.pi / N * 0.8

# ---------------------- 画图 ----------------------
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

bars = ax.bar(
    theta,
    sizes,
    width=width,
    bottom=0,
    edgecolor='white',
    linewidth=1
)

# ---------------------- 颜色（保持你统一色调） ----------------------
for bar in bars:
    bar.set_facecolor('#D8E6E7')

# ---------------------- 标注数值 ----------------------
for angle, radius in zip(theta, sizes):
    ax.text(
        angle,
        radius + max(sizes)*0.05,
        f'{radius:,}',
        ha='center',
        va='center',
        fontsize=10
    )

# ---------------------- 设置标签 ----------------------
ax.set_xticks(theta)
ax.set_xticklabels(labels)

# ---------------------- 美化 ----------------------
ax.set_yticklabels([])  # 隐藏径向刻度（更干净）
ax.spines['polar'].set_visible(False)

# 让0°在顶部，顺时针
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# ---------------------- 总数 ----------------------
total = sum(sizes)
plt.title(f'{total:,} WSIs', fontsize=FONTSIZE, pad=20)

# ---------------------- 保存 ----------------------
plt.tight_layout()
plt.savefig(f'PLOTS/0.tableone/radial_bar_{plot_case}.svg', dpi=300, bbox_inches='tight')
plt.show()
plt.close()