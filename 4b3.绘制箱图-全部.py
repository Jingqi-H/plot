import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 18

# =========================
# 文件路径
# =========================
save_root = f'PLOTS/4.联合建模/svg'
os.makedirs(save_root, exist_ok=True)

files = ['SXCH-Train', 'SXCH-Val', 'All_external', 'YYH', 'SYSUCC', 'TCGA_STAD']
for center in files:
    if center == 'TCGA_STAD':
        continue
    
    cindex_csv = f'PLOTS/4.联合建模/cox_cindex_{center}_matrix.csv'
    save_path = f'{save_root}/{center}_cindexboxplot.svg'

    cindex_df = pd.read_csv(cindex_csv)

    # =========================
    # 绘制箱线图
    # =========================
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9,8))

    # 每个箱子设置颜色
    # box_colors = [color_dict.get(col, 'lightgray') for col in cindex_df.columns]
    # 设置颜色为渐变蓝色， 由浅到深
    box_colors = plt.cm.Blues(np.linspace(0.1, 0.7, len(cindex_df.columns)))


    # 箱线图
    # 设置线宽为2
    bp = ax.boxplot(
        [cindex_df[col] for col in cindex_df.columns],
        labels=cindex_df.columns,
        patch_artist=True,  # 可以填充颜色
        medianprops=dict(color='black', linewidth=2),  # 中位数线：黑色 + 线宽2
        boxprops=dict(linewidth=2),                    # 箱体边框线宽2
        whiskerprops=dict(linewidth=2),                # 须（上下延伸线）线宽2
        capprops=dict(linewidth=2),                    # 须两端横线（帽）线宽2
        widths=0.5,              # 控制箱子宽度
        showfliers=False,         # 显示异常值散点,有扰动
    )

    # 设置颜色
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)

    # 删除上边界和右边界
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["left"].set_linewidth(1.8)

    # 其他美化
    if center == 'All_external':
        center = 'External'
    ax.set_title(center)

    ax.set_ylabel('CIndex')
    # ax.set_xticklabels(cindex_df.columns, rotation=90, ha='center')

    # 设置y轴固定为0.5-0.8， 间隔0.05
    ax.set_ylim(bottom=0.48,top=0.81)
    # 2. 设置y轴刻度数字：从0.50开始，步长0.05，到0.80结束（仅显示这些数字）
    ax.set_yticks(np.arange(0.50, 0.81, 0.05))
    
    ax.set_xticklabels(cindex_df.columns, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()