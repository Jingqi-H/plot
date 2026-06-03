import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


# ======================
# 读取数据
# ======================

df = pd.read_excel("PLOTS/0.tableone/[all_cohort] 画pie的表格.xlsx")

train_col = "SXCH Training (n=3707), No. (%)"
val_col = "SXCH Validation (n=3707), No. (%)"

# ======================
# 提取 count
# ======================

def extract_count(x):
    if pd.isna(x):
        return 0
    return int(str(x).split(" ")[0])

df["train_n"] = df[train_col].apply(extract_count)
df["val_n"] = df[val_col].apply(extract_count)

df["count"] = df["train_n"] + df["val_n"]

# ======================
# 需要画的变量
# ======================

variables = [
    "OS_status",
    "Lauren Type",
    "Grade",
    "Histo Type",
    "Stage"
]

# ======================
# 渐变色函数
# ======================

def gradient_colors(base_color, n):

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "grad", ["white", base_color]
    )

    colors = [cmap(i/(n+1)) for i in range(1, n+1)]
    return colors

# ======================
# 颜色设置
# ======================

base_colors = {
    "OS_status": "#4C72B0",
    "Lauren Type": "#55A868",
    "Grade": "#C44E52",
    "Histo Type": "#8172B2",
    "Stage": "#CCB974"
}

# ======================
# 开始绘图
# ======================

fig, ax = plt.subplots(figsize=(10,10))


radius = 0.5
width = 0.15

for i, var in enumerate(variables):

    sub = df[df["Variable"] == var]

    counts = sub["count"].values
    labels = sub["type"].values

    # 生成渐变色
    colors = gradient_colors(base_colors[var], len(counts))

    # Missing 设置为灰色
    for j,l in enumerate(labels):
        if str(l).lower() == "missing":
            colors[j] = "lightgray"

    ax.pie(
        counts,
        radius=radius + i*width,
        labels=None,
        colors=colors,
        wedgeprops=dict(width=width, edgecolor='white')
    )

# 创建 legend patches
legend_patches = [
    mpatches.Patch(color=color, label=label)
    for label, color in base_colors.items()
]

# 添加 legend
ax.legend(
    handles=legend_patches,
    title="Variables",
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    frameon=False
)

# 中心空心
# centre_circle = plt.Circle((0,0),0.3,fc='white')
centre_circle = plt.Circle((0,0),0.3,fc='white')
fig.gca().add_artist(centre_circle)

plt.title("Distribution of Clinical Variables", fontsize=16)

plt.show()