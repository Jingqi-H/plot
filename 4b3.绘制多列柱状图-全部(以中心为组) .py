import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

FONT_SIZE = 18
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = FONT_SIZE

# =========================
# 路径
# =========================
save_root = f'PLOTS/4.联合建模/svg'
os.makedirs(save_root, exist_ok=True)

centers = ['SXCH-Train', 'SXCH-Val', 'External', 'CMU1H','YYH', 'SYSUCC', 'TCGA_STAD']
models = ['Age', 'CEA', 'Location', 'pT', 'pN', 'pM', 'Stage', 'Clinical', 'GRASP', 'Clinical+GRASP']

# =========================
# 配色（现在是“模型配色”）
# =========================
# color_df = pd.read_excel('PLOTS/@source/color scheme (Covariate).xlsx')
# 用sns设置渐变的蓝色
colors = plt.cm.Blues(np.linspace(0, 1, len(models)))

# =========================
# 统计
# =========================
mean_dict = {m: [] for m in models}
ci_dict = {m: [] for m in models}

clinical_vals = {}
grasp_vals = {}

for center in centers:
    cindex_csv = f'PLOTS/4.联合建模/cox_cindex_{center}_matrix.csv'
    df = pd.read_csv(cindex_csv)

    clinical_vals[center] = df['Clinical'].values
    grasp_vals[center] = df['Clinical+GRASP'].values

    for m in models:
        if m not in df.columns:
            mean_dict[m].append(np.nan)
            ci_dict[m].append([np.nan, np.nan])
            continue

        vals = df[m].dropna().values

        if len(vals) == 0:
            mean_dict[m].append(np.nan)
            ci_dict[m].append([np.nan, np.nan])
            continue

        mean = np.mean(vals)
        lower = np.percentile(vals, 2.5)
        upper = np.percentile(vals, 97.5)

        mean_dict[m].append(mean)
        ci_dict[m].append([mean - lower, upper - mean])

# =========================
# 绘图（核心修改在这里）
# =========================
n_models = len(models)
n_centers = len(centers)

x = np.arange(n_centers)   # ⭐ x轴变成 center
bar_width = 0.08           # 模型多 → bar更窄

plt.figure(figsize=(22, 8))

for j, m in enumerate(models):   # ⭐ 外层循环变成 model
    means = np.array(mean_dict[m])
    cis = np.array(ci_dict[m])

    valid = ~np.isnan(means)
    # color = color_df[color_df['Covariates'] == m]['Colors'].values[0]

    bars = plt.bar(
        x[valid] + j * bar_width,
        means[valid],
        width=bar_width,
        yerr=cis[valid].T,
        capsize=2,
        color=colors[j],
        label=m,
        edgecolor='black',
        linewidth=0.4
    )

    # 数值标注
    
    valid_idx = np.where(valid)[0]
    for idx, bar in zip(valid_idx, bars):
        height = bar.get_height()
        
        color = 'white' if m == 'Clinical+GRASP' else 'black'
        plt.text(
            bar.get_x() + bar.get_width()/2,
            0.02,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
            rotation=90,
            color=color
        )
    if m == 'CEA':
        offset = j * bar_width
        missing_idx = 5   # 第二个方法(0开始)

        x_na = x[missing_idx] + offset

        plt.text(
            x_na,
            0.02,
            'NA',
            ha='center',
            va='bottom',
            fontsize=9,
            rotation=90,
            color='black'
        )

# =========================
# 显著性（按center做）
# =========================
clinical_idx = models.index('Clinical')
combo_idx = models.index('Clinical+GRASP')

for i, center in enumerate(centers):
    vals1 = clinical_vals[center]
    vals2 = grasp_vals[center]

    stat, p = wilcoxon(vals1, vals2)

    if p < 0.0001:
        star = '****'
    elif p < 0.001:
        star = '***'
    elif p < 0.01:
        star = '**'
    elif p < 0.05:
        star = '*'
    else:
        star = 'ns'

    x1 = i + clinical_idx * bar_width
    x2 = i + combo_idx * bar_width

    # y = max(mean_dict['Clinical'][i], mean_dict['Clinical+GRASP'][i]) + 0.02
    c_means = np.array(mean_dict['Clinical'])  
    c_cis = np.array(ci_dict['Clinical'])
    valid = ~np.isnan(c_means)
    cc_means = np.array(mean_dict['Clinical+GRASP'])  
    cc_cis = np.array(ci_dict['Clinical+GRASP'])
    valid = ~np.isnan(cc_means)
    y = max(c_means[i]+c_cis[i][1], cc_means[i]+cc_cis[i][1]) + 0.005

    plt.plot([x1, x1, x2, x2], [y, y+0.005, y+0.005, y], lw=1, c='black')
    plt.text((x1+x2)/2, y+0.001, star, ha='center', va='bottom', fontsize=10)

# =========================
# 美化
# =========================
plt.xticks(x + bar_width*(n_models-1)/2, centers, rotation=0)
plt.ylabel('C-index')

# plt.legend(
#     frameon=False,
#     bbox_to_anchor=(1.02, 1),
#     loc='upper left',
#     fontsize=FONT_SIZE-4
# )
plt.legend(
    frameon=False,
    loc='lower left',          # 以图上方为参考点
    bbox_to_anchor=(0.001, 0.945),  # 水平居中，放在上面
    ncol=10,                      # 横着排（根据你的模型数量改）
    fontsize=FONT_SIZE-5
)

plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

save_path = f'{save_root}/summary_barplot_by_center.svg'
plt.savefig(save_path)
plt.show()