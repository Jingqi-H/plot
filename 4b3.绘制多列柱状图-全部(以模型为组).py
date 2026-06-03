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

centers = ['SXCH-Train', 'SXCH-Val', 'External', 'YYH', 'SYSUCC', 'TCGA_STAD']
models = ['Age', 'CEA', 'Location', 'pT', 'pN', 'pM', 'Stage', 'Clinical', 'GRASP', 'Clinical+GRASP']

# =========================
# 期刊级配色（colorblind-friendly）
# =========================
colors = [
    "#f1b4b0",  # blue
    "#aadddd",  # orange
    "#fae4d1",  # green
    "#c9d7e5",  # red
    "#bdbcdb",  # purple
    '#d7e9cd'
]

# =========================
# 统计
# =========================
mean_dict = {m: [] for m in models}
ci_dict = {m: [] for m in models}

# 存储用于显著性检验
clinical_vals = {}
grasp_vals = {}

for center in centers:
    # if center != 'TCGA_STAD':
    #     continue
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
# 绘图
# =========================
n_models = len(models)
n_centers = len(centers)

x = np.arange(n_models)
bar_width = 0.15

plt.figure(figsize=(20, 8))

for i, center in enumerate(centers):
    
    means = np.array([mean_dict[m][i] for m in models])
    cis = np.array([ci_dict[m][i] for m in models])

    valid = ~np.isnan(means)

    bars = plt.bar(
        x[valid] + i * bar_width,
        means[valid],
        width=bar_width,
        yerr=cis[valid].T,
        capsize=3,
        color=colors[i],
        label=center,
        edgecolor='black',
        linewidth=0.5
    )

    valid_idx = np.where(valid)[0]

    for idx, bar in zip(valid_idx, bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            0.02,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=11,
            color='black',
            rotation=90
        )

# =========================
# 显著性标注（Clinical vs Clinical+GRASP）
# =========================
clinical_idx = models.index('Clinical')
combo_idx = models.index('Clinical+GRASP')
for i, center in enumerate(centers):
    x1 = clinical_idx + i * bar_width
    x2 = combo_idx + i * bar_width

    vals1 = clinical_vals[center]
    vals2 = grasp_vals[center]

    # 配对检验（bootstrap推荐）
    stat, p = wilcoxon(vals1, vals2)

    # 星号规则
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

    # y = max(mean_dict['Clinical'][i], mean_dict['Clinical+GRASP'][i]) + 0.02
    c_means = np.array(mean_dict['Clinical'])  
    c_cis = np.array(ci_dict['Clinical'])
    valid = ~np.isnan(c_means)
    cc_means = np.array(mean_dict['Clinical+GRASP'])  
    cc_cis = np.array(ci_dict['Clinical+GRASP'])
    valid = ~np.isnan(cc_means)
    y = max(c_means[i]+c_cis[i][1], cc_means[i]+cc_cis[i][1]) + 0.005

    plt.plot([x1, x1, x2, x2], [y, y+0.005, y+0.005, y], lw=1.2, c='black')
    plt.text((x1+x2)/2, y+0.001, star, ha='center', va='bottom', fontsize=10)

# =========================
# 美化
# =========================
plt.xticks(x + bar_width*(n_centers-1)/2, models, rotation=0)
plt.ylabel('C-index')

# plt.ylim(0.5, 0.85)

# plt.legend(frameon=False, loc='upper left', fontsize=FONT_SIZE-2)
plt.legend(
    frameon=False,
    loc='lower left',          # 以图上方为参考点
    bbox_to_anchor=(0.001, 0.945),  # 水平居中，放在上面
    ncol=10,                      # 横着排（根据你的模型数量改）
    fontsize=FONT_SIZE-5
)

plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

save_path = f'{save_root}/summary_barplot.svg'
plt.savefig(save_path)
plt.show()



# =========================
# 保存结果到 Excel
# =========================
results = []

for i, center in enumerate(centers):
    for m in models:
        if m == 'Stage':
            print(center, m, mean_dict[m][i], ci_dict[m][i])
        mean = mean_dict[m][i]
        lower = mean - ci_dict[m][i][0]
        upper = mean + ci_dict[m][i][1]

        # 默认没有p值
        p_val = np.nan

        # 只对 Clinical vs Clinical+GRASP 计算p值
        if m == 'Clinical+GRASP':
            vals1 = clinical_vals[center]
            vals2 = grasp_vals[center]

            stat, p_val = wilcoxon(vals1, vals2)

        results.append({
            'Center': center,
            'Model': m,
            'Mean C-index': round(mean, 4),
            'Lower 95% CI': round(lower, 4),
            'Upper 95% CI': round(upper, 4),
            'P-value (vs Clinical)': p_val
        })

# 转成 DataFrame
results_df = pd.DataFrame(results)

# =========================
# 额外格式：合并成一列（论文常用）
# =========================
# results_df['C-index (95% CI)'] = (
#     results_df['Mean C-index'].astype(str) + ' (' +
#     results_df['Lower 95% CI'].astype(str) + '–' +
#     results_df['Upper 95% CI'].astype(str) + ')'
# )
results_df['C-index (95% CI)'] = results_df.apply(
    lambda row: (
        f"{row['Mean C-index']:.4f} ({row['Lower 95% CI']:.4f}–{row['Upper 95% CI']:.4f})"
        if not np.isnan(row['Mean C-index']) else "-"
    ),
    axis=1
)

# =========================
# 保存 Excel
# =========================
excel_path = f'{save_root}/summary_cindex.xlsx'

with pd.ExcelWriter(excel_path) as writer:
    # Sheet 1：标准格式
    results_df.to_excel(writer, sheet_name='long_format', index=False)

    # Sheet 2：宽表（更像论文Table）
    pivot_df = results_df.pivot(index='Model', columns='Center', values='C-index (95% CI)')
    # pivot_df = pivot_df.loc[models]
    pivot_df = pivot_df.reindex(models)
    pivot_df = pivot_df[centers]
    pivot_df.to_excel(writer, sheet_name='table_format')

print(f'Saved to: {excel_path}')