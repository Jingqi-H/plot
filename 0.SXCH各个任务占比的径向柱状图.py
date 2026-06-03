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


def extract_count(x):
    if pd.isna(x):
        return 0
    return int(str(x).split(" ")[0])


df["train_n"] = df[train_col].apply(extract_count)
df["val_n"] = df[val_col].apply(extract_count)

df["count"] = df["train_n"] + df["val_n"]

# ======================
# Variable顺序
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

    return [cmap(i/(n+1)) for i in range(1, n+1)]

# ======================
# base colors
# ======================

base_colors = {
    "OS_status": "#4C72B0",
    "Lauren Type": "#55A868",
    "Grade": "#C44E52",
    "Histo Type": "#8172B2",
    "Stage": "#CCB974"
}

# ======================
# 准备数据
# ======================

plot_data = []

for var in variables:

    sub = df[df["Variable"] == var]

    labels = sub["type"].values
    counts = sub["count"].values

    colors = gradient_colors(base_colors[var], len(counts))

    for j, l in enumerate(labels):

        if str(l).lower() == "missing":
            colors[j] = "lightgray"

        plot_data.append({
            "Variable": var,
            "Type": l,
            "Count": counts[j],
            "Color": colors[j]
        })

plot_df = pd.DataFrame(plot_data)
plot_df["RealCount"] = plot_df["Count"]
plot_df["Count"] = np.log10(plot_df["Count"] + 40)

# ======================
# 极坐标绘图
# ======================

fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111, polar=True)

N = len(plot_df)

angles = np.linspace(0, 2*np.pi, N, endpoint=False)

max_count = plot_df["Count"].max()

inner_radius = max_count * 0.4  # 控制中间空白大小

bars = ax.bar(
    angles,
    plot_df["Count"],
    width=2*np.pi/N*0.9,
    # bottom=inner_radius,
    bottom=0,
    color=plot_df["Color"],
    edgecolor="white"
)


ax.set_ylim(0, inner_radius + plot_df["Count"].max())
centre_circle = plt.Circle(
    (0,0),
    inner_radius,
    transform=ax.transData._b,
    color='white',
    zorder=10
)
# 将他添加到画布
ax.add_artist(centre_circle)


zero_radius = 0  # 柱子0位置的半径（极坐标原点）
zero_circle = plt.Circle(
    (0, 0),
    inner_radius,
    transform=ax.transData._b,  # 与原有中心圆保持一致的transform
    facecolor='none',           # 无填充色（仅边框）
    edgecolor='gray',          # 边框颜色（可改为gray、darkgray等）
    linewidth=1,                # 边框粗细（可调整）
    # linestyle='--',
    zorder=11                   # 层级高于中心圆，确保可见
)
ax.add_artist(zero_circle)

# ======================
# 新增：绘制每个柱子的切向文字（RealCount）
# ======================
# 文字偏移量（柱子顶端上方，避免重叠）
text_offset = max_count * 0.02  

for j in range(N):
    # 1. 获取当前柱子的关键参数
    bar_angle = angles[j]  # 柱子的角度
    bar_height = plot_df["Count"].iloc[j]  # 柱子的高度（对数后）
    real_count = plot_df["RealCount"].iloc[j]  # 要显示的真实数值
    
    # 2. 计算文字的位置（极坐标）
    # 半径 = 柱子高度 + 偏移量（在柱子顶端上方）
    text_radius = bar_height + text_offset
    # 极坐标转直角坐标（用于文字定位，可选，直接用极坐标更简单）
    x = text_radius * np.cos(bar_angle)
    y = text_radius * np.sin(bar_angle)
    
    # 3. 计算文字旋转角度（关键：切向、与圆弧一致、和柱子90°）
    # 适配theta_offset（pi/2）和theta_direction（-1，顺时针）
    rotation = np.degrees(bar_angle) - 90  # 基础旋转
    # 调整文字朝向（避免倒字）
    if rotation > 90 and rotation < 270:
        rotation += 180  # 翻转180°，保证文字正读
    
    # 4. 绘制切向文字
    ax.text(
        bar_angle,          # 文字的角度（极坐标）
        text_radius,        # 文字的半径（极坐标）
        f"{int(real_count)}",  # 显示RealCount（取整）
        rotation=0,  # 切向旋转角度
        ha='center',        # 水平居中
        va='center',        # 垂直居中
        fontsize=8,         # 字体大小（可调整）
        color='black',      # 文字颜色
        fontweight='bold',  # 加粗，提升可读性
        zorder=12           # 层级最高，避免被遮挡
    )


# ======================
# 美化
# ======================
# ax.grid(False)          # 去掉所有参考网格
ax.spines['polar'].set_visible(False)  # 去掉外圈边框

ax.set_theta_offset(np.pi / 2)  # 正上方为0度
ax.set_theta_direction(-1) # 顺时针

ax.set_xticks([])
ax.set_yticks([])

# ======================
# legend
# ======================

legend_patches = [
    mpatches.Patch(color=color, label=label)
    for label, color in base_colors.items()
]

ax.legend(
    handles=legend_patches,
    title="Variables",
    bbox_to_anchor=(1.1,0.5),
    loc="center left",
    frameon=False
)
plt.savefig("PLOTS/0.tableone/0.SXCH各个任务占比的径向柱状图.svg", dpi=300, bbox_inches='tight')

plt.show()