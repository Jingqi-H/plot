'训练集做单多因素分析，得到具有显著性的因素'

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from matplotlib.gridspec import GridSpec



def merge_uni_multi_cox(
    uni_path,
    multi_path,
    save_path,
    uni_sheet="Univariate_Cox",
    multi_sheet="Multivariate_Cox"
):

    # =========================
    # 1️⃣ 读取
    # =========================
    uni_df = pd.read_excel(uni_path, sheet_name=uni_sheet)
    multi_df = pd.read_excel(multi_path, sheet_name=multi_sheet)

    # 统一变量列名
    if "Variable" in uni_df.columns:
        uni_df = uni_df.rename(columns={"Variable": "Covariate"})
    if "Variable" in multi_df.columns:
        multi_df = multi_df.rename(columns={"Variable": "Covariate"})

    # =========================
    # 2️⃣ 提前重命名（防止_x/_y）
    # =========================
    uni_df = uni_df.rename(columns={
        "HR": "Uni_HR",
        "95% CI": "Uni_CI",
        "P value": "Uni_P"
    })

    multi_df = multi_df.rename(columns={
        "HR": "Multi_HR",
        "95% CI": "Multi_CI",
        "P value": "Multi_P"
    })

    # =========================
    # 3️⃣ merge
    # =========================
    merged = pd.merge(
        uni_df[["Covariate", "Uni_HR", "Uni_CI", "Uni_P"]],
        multi_df[["Covariate", "Multi_HR", "Multi_CI", "Multi_P"]],
        on="Covariate",
        how="left"   # 🔥 关键：以单因素为主
    )

    # =========================
    # 4️⃣ 构造 HR (95% CI)
    # =========================
    merged["Univariate HR (95% CI)"] = (
        merged["Uni_HR"].round(2).astype(str)
        + " ("
        + merged["Uni_CI"]
        + ")"
    )

    merged["Multivariate HR (95% CI)"] = (
        merged["Multi_HR"].round(2).astype(str)
        + " ("
        + merged["Multi_CI"]
        + ")"
    )

    # =========================
    # 5️⃣ 生成最终表
    # =========================
    df_final = pd.DataFrame({
        "Covariate": merged["Covariate"],
        "Univariate HR (95% CI)": merged["Univariate HR (95% CI)"],
        "P (Uni)": merged["Uni_P"],
        "Multivariate HR (95% CI)": merged["Multivariate HR (95% CI)"],
        "P (Multi)": merged["Multi_P"],
    })

    # =========================
    # 6️⃣ 处理多因素空值
    # =========================
    df_final["Multivariate HR (95% CI)"] = df_final["Multivariate HR (95% CI)"].fillna("—")
    df_final["P (Multi)"] = df_final["P (Multi)"].fillna("—")

    # =========================
    # 7️⃣ 保存
    # =========================
    df_final.to_excel(save_path, index=False)

    print("✅ 合并完成：", save_path)

    return df_final

# ============================================================
#  构建 Cox 用多因素数据
# ============================================================
def prepare_multivariate_df(df, exclude_cols=['case_id', 'OS']):
    """
    1 连续变量二值化
    2 TNM 分组合并
    3 One-hot 编码分类变量
    """
    cox_df = df.copy()

    cox_df['Age'] = (cox_df['Age'] > 65).astype(int)
    num_cols = cox_df.select_dtypes(include=[np.number]).columns
    # 需要转换的列 = 数值列 - 排除列
    cols_to_convert = [col for col in num_cols if col not in exclude_cols]
    # 转换为 pandas 可空整数类型（不会因为 NaN 报错）
    cox_df[cols_to_convert] = cox_df[cols_to_convert].astype('Int64')


    cox_df = pd.get_dummies(
        cox_df,
        columns=['Age', 'pT', 'pN',  'Gender', 'Location',  'HistologicalType',
       'LaurenType', 'Chemotherapy', 'Stage', 'Grade'],
        drop_first=True
    )

    return cox_df

# ============================================================
#  合并 Cox 临床数据 + k 折预测结果
# ============================================================
def merge_data(
    info_df,
    result_df,
    cutoff,
):
    """
    1 合并 slide → patient
    2 聚合 risk score
    3 构造 survival_time / status / risk_group
    """
    
    

    result_df['event_times'] = result_df['survival_time'].copy()
    result_df['status'] = 1 - result_df['censorship']
    result_df['risk_group'] = (result_df['risk'] > cutoff).astype(int)

    final_df = pd.merge(
        result_df[['case_id', 'slide_id', 'slide_id_wax', 'risk', 'censorship', 'survival_time', 'event_times', 'status', 'risk_group']],
        info_df,
        on='case_id',
        how='inner'
    )

    # final_df.drop(columns=['case_id'], inplace=True)
    final_df.dropna(subset=['survival_time', 'status'], inplace=True)

    return final_df


# ============================================================
#  Cox 森林图
# ============================================================
def plot_cox_forest(
    cph,
    save_path,
    rename_dict=None,
    descending=True,
    fontsize=12,
):
    """
    1 提取 HR 和 CI
    2 log2 坐标森林图
    """


    summary = cph.summary.copy()
    if rename_dict is not None:
        summary = summary.rename(index=rename_dict)
        ordered_labels = list(rename_dict.values())
        summary = summary.loc[ordered_labels]
    summary['HR'] = np.exp(summary['coef'])
    summary['CI_low'] = np.exp(summary['coef lower 95%'])
    summary['CI_high'] = np.exp(summary['coef upper 95%'])

    # summary = summary.sort_values('HR', ascending=not descending)

    hr = summary['HR'].values
    ci_low = summary['CI_low'].values
    ci_high = summary['CI_high'].values
    labels = summary.index.tolist()

    y_pos = np.arange(len(hr))[::-1]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.7 * len(hr))))

    ax.hlines(y_pos, ci_low, ci_high, linewidth=2,color='black',)
    ax.plot(hr, y_pos, 'o', markersize=8,color='black')
    ax.axvline(1, linestyle='--', color='gray')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Hazard Ratio (HR)', fontsize=fontsize)

    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()




def plot_cox_forest_table(
    cph,
    save_path,
    rename_dict=None,
    fontsize=12,
):

    summary = cph.summary.copy()

    # rename + 顺序
    if rename_dict is not None:
        summary = summary.rename(index=rename_dict)
        ordered_labels = list(rename_dict.values())
        summary = summary.loc[ordered_labels]

    # 计算 HR 和 CI
    summary["HR"] = np.exp(summary["coef"])
    summary["CI_low"] = np.exp(summary["coef lower 95%"])
    summary["CI_high"] = np.exp(summary["coef upper 95%"])

    labels = summary.index.tolist()
    hr = summary["HR"].values
    ci_low = summary["CI_low"].values
    ci_high = summary["CI_high"].values
    pvals = summary["p"].values

    n = len(labels)
    y_pos = np.arange(n)[::-1]

    # ------------------------------------------------
    # 文本格式
    # ------------------------------------------------
    hr_text = [f"{x:.2f}" for x in hr]
    ci_text = [f"({l:.2f}–{h:.2f})" for l, h in zip(ci_low, ci_high)]

    p_text = []
    for p in pvals:
        if p < 0.05:
            p_text.append("<0.05")
        else:
            p_text.append(f"{p:.3f}")

    # ------------------------------------------------
    # Figure layout
    # ------------------------------------------------
    fig = plt.figure(figsize=(11, 13))

    gs = GridSpec(
        1,
        5,
        width_ratios=[3, 1, 2, 1.5, 4],
        wspace=0.05
    )

    ax_label = fig.add_subplot(gs[0, 0])
    ax_hr = fig.add_subplot(gs[0, 1])
    ax_ci = fig.add_subplot(gs[0, 2])
    ax_p = fig.add_subplot(gs[0, 3])
    ax_forest = fig.add_subplot(gs[0, 4])

    # ------------------------------------------------
    # Label
    # ------------------------------------------------
    ax_label.set_ylim(-1, n)
    for i, label in enumerate(labels[::-1]):
        # ax_label.text(0, i, label, fontsize=fontsize, va="center")
        ax_label.text(0.5, i, label, ha="center", va="center", fontsize=fontsize)
    ax_label.set_title("Variable", fontsize=fontsize)
    ax_label.axis("off")

    # ------------------------------------------------
    # HR column
    # ------------------------------------------------
    ax_hr.set_ylim(-1, n)
    for i, txt in enumerate(hr_text[::-1]):
        ax_hr.text(0.5, i, txt, ha="center", va="center", fontsize=fontsize)

    ax_hr.set_title("HR", fontsize=fontsize)
    ax_hr.axis("off")

    # ------------------------------------------------
    # CI column
    # ------------------------------------------------
    ax_ci.set_ylim(-1, n)
    for i, txt in enumerate(ci_text[::-1]):
        ax_ci.text(0.5, i, txt, ha="center", va="center", fontsize=fontsize)

    ax_ci.set_title("95% CI", fontsize=fontsize)
    ax_ci.axis("off")

    # ------------------------------------------------
    # P column
    # ------------------------------------------------
    ax_p.set_ylim(-1, n)
    for i, txt in enumerate(p_text[::-1]):
        ax_p.text(0.5, i, txt, ha="center", va="center", fontsize=fontsize)

    ax_p.set_title("P value", fontsize=fontsize)
    ax_p.axis("off")

    # ------------------------------------------------
    # Forest plot
    # ------------------------------------------------
    ax_forest.hlines(
        y_pos,
        ci_low,
        ci_high,
        color="black",
        # linewidth=2,
    )

    # ax_forest.axvline(1, linestyle="--", color="gray")
    ax_forest.axvline(1, linestyle="--", color="gray")


    ax_forest.plot(
        hr,
        y_pos,
        "o",
        markerfacecolor="#5285BD",  # 填充色设置为指定的蓝色
        # markerfacecolor="black",  # 填充色设置为指定的蓝色
        markeredgecolor="black",     # 边框设置为黑色
        # markeredgewidth=2,         # 边框宽度（可选，增强视觉效果）
        markersize=10,
    )



    ax_forest.set_ylim(-1, n)
    ax_forest.set_yticks([])

    ax_forest.set_xscale("log", base=2)
    ax_forest.set_xlabel("Hazard Ratio", fontsize=fontsize)

    ax_forest.tick_params(axis='x',labelsize=fontsize)

    ax_forest.grid(axis="x", linestyle="--", alpha=0.3)

    # 去掉上右边框
    ax_forest.spines["top"].set_visible(False)
    ax_forest.spines["right"].set_visible(False)
    ax_forest.spines["left"].set_visible(False)
    # ax_forest.spines["bottom"].set_linewidth(2)

    # hline_y_positions = [n+1, n, -1]
    # # 2. 汇总所有子图，给每个子图画横线（拼接后横跨整个画布）
    # all_axes = [ax_label, ax_hr, ax_ci, ax_p, ax_forest]
    # # 3. 遍历绘制横线
    # for y_pos_hline in hline_y_positions:
    #     for ax in all_axes:
    #         ax.axhline(
    #             y=y_pos_hline,    # 横线Y坐标
    #             xmin=0,           # 横线起始：子图最左侧
    #             xmax=1,           # 横线结束：子图最右侧
    #             color='black',    # 颜色
    #             linewidth=1.5,    # 线宽
    #             linestyle='-'     # 线型：实线
    #         )


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()





def save_univariate_cox_results(
    uni_results_df,
    save_path,
    rename_dict=None,
):
    """
    保存单因素 Cox 结果到 Excel
    Sheet1: 原始结果
    Sheet2: 论文格式（Variable + HR + 95% CI + P value）
    """

    # =====================
    # Sheet 1：原始结果
    # =====================
    raw_summary = uni_results_df.copy()
    raw_summary.index.name = 'index'

    # =====================
    # Sheet 2：论文格式整理
    # =====================
    summary = raw_summary.copy()

    summary['HR'] = summary['exp(coef)']
    summary['HR_lower'] = summary['exp(coef) lower 95%']
    summary['HR_upper'] = summary['exp(coef) upper 95%']

    paper_df = pd.DataFrame({
        'Variable': summary['Variable'],
        'HR': summary['HR'].round(4),
        '95% CI': summary['HR_lower'].round(2).astype(str)
                  + '–'
                  + summary['HR_upper'].round(2).astype(str),
        'P value': summary['p']
    })

    # 变量重命名（可选）
    if rename_dict is not None:
        paper_df['Variable'] = paper_df['Variable'].map(
            lambda x: rename_dict.get(x, x)
        )

    # P值格式
    paper_df['P value'] = paper_df['P value'].apply(
        lambda x: f'{x:.2g}' if x >= 0.05 else '<0.05'
    )

    # =====================
    # 写 Excel
    # =====================
    with pd.ExcelWriter(save_path) as writer:
        raw_summary.to_excel(writer, sheet_name='Raw_Univariate_Cox')
        paper_df.to_excel(writer, sheet_name='Univariate_Cox', index=False)

    print(f"单因素Cox结果已保存至: {save_path}")

def run_univariate_cox(final_df, varList):
    
    

    print('单因素变量：', varList)

    cph = CoxPHFitter(penalizer=0.1)
    uni_results = []

    for var in varList:

        tempt_df = final_df[['survival_time','status',var]].dropna().copy()

        cph.fit(
            tempt_df,
            duration_col='survival_time',
            event_col='status',
            formula=var
        )

        summary = cph.summary.copy()
        summary['Covariate'] = var
        summary['significant'] = summary['p'] < 0.05
        # summary['significant'] = summary['p'] < 0.001

        uni_results.append(summary)

        hr = summary.loc[var, 'exp(coef)']
        hr_lower = summary.loc[var, 'exp(coef) lower 95%']
        hr_upper = summary.loc[var, 'exp(coef) upper 95%']
        p_value = summary.loc[var, 'p']

        print(f"{var}: HR={hr:.3f} ({hr_lower:.3f}-{hr_upper:.3f}), p={p_value:.2e}")

    return pd.concat(uni_results)


def run_multivariate_cox(final_df, covariates):

    cph = CoxPHFitter(penalizer=0.1)

    cph.fit(
        final_df[['survival_time', 'status'] + covariates],
        duration_col='survival_time',
        event_col='status'
    )

    return cph

def save_univariate_cox_results(
    uni_results_df,
    save_path,
    rename_dict=None,
):

    raw_summary = uni_results_df.copy()

    summary = raw_summary.copy()
    summary['HR'] = summary['exp(coef)']
    summary['HR_lower'] = summary['exp(coef) lower 95%']
    summary['HR_upper'] = summary['exp(coef) upper 95%']

    paper_df = pd.DataFrame({
        'Covariate': summary['Covariate'],
        'HR': summary['HR'].round(4),
        '95% CI': summary['HR_lower'].round(2).astype(str)
                  + '–'
                  + summary['HR_upper'].round(2).astype(str),
        'P value': summary['p']
    })

    if rename_dict is not None:

        # 重命名
        paper_df['Covariate'] = paper_df['Covariate'].map(
            lambda x: rename_dict.get(x, x)
        )

        # 🔥 强制顺序按 rename_dict
        ordered_vars = [rename_dict[k] for k in rename_dict if k in summary['Covariate'].values]
        paper_df = paper_df.set_index('Covariate').loc[ordered_vars].reset_index()

    paper_df['P value'] = paper_df['P value'].apply(
        lambda x: f'{x:.2g}' if x >= 0.05 else '<0.05'
        # lambda x: f'{x:.2g}' if x >= 0.001 else '<0.001'
    )

    with pd.ExcelWriter(save_path) as writer:
        raw_summary.to_excel(writer, sheet_name='Raw_Univariate_Cox')
        paper_df.to_excel(writer, sheet_name='Univariate_Cox', index=False)


def save_multivariate_cox_results(
    summary,
    save_path,
    rename_dict=None,
):

    raw_summary = summary.copy()

    summary = raw_summary.copy()

    summary['HR'] = summary['exp(coef)']
    summary['CI_low'] = summary['exp(coef) lower 95%']
    summary['CI_high'] = summary['exp(coef) upper 95%']

    paper_df = pd.DataFrame({
        'Covariate': summary.index,
        'HR': summary['HR'].round(4),
        '95% CI': summary['CI_low'].round(2).astype(str)
                  + '–'
                  + summary['CI_high'].round(2).astype(str),
        'P value': summary['p']
    })

    if rename_dict is not None:

        paper_df['Covariate'] = paper_df['Covariate'].map(
            lambda x: rename_dict.get(x, x)
        )

        ordered_vars = [rename_dict[k] for k in rename_dict if k in summary.index]
        paper_df = paper_df.set_index('Covariate').loc[ordered_vars].reset_index()

    paper_df['P value'] = paper_df['P value'].apply(
        lambda x: f'{x:.2g}' if x >= 0.05 else '<0.05'
    )

    with pd.ExcelWriter(save_path) as writer:
        raw_summary.to_excel(writer, sheet_name='Raw_Cox_Summary')
        paper_df.to_excel(writer, sheet_name='Multivariate_Cox', index=False)

exp_id = '0.0001loadloss'

RESULT_DIR = f"results/moe_wsi/{exp_id}"
INFO_PATH = "ori_files/SXCH/clinical_info_all.csv"
CUTOFF_PATH = f"PLOTS/2.km/{exp_id}/cutoff_media_risk_score.csv"


# ---------- 1 读取并预处理临床数据 ----------
raw_df = pd.read_csv(INFO_PATH)
raw_df.drop_duplicates('case_id', inplace=True)
raw_df = raw_df[['case_id', 'Age', 'Gender', 'Location', 'Grade', 'Histo Type',
       'Lauren Type', 'Chemotherapy', 'Stage', 'Pathological T stage', 'Pathological N stage', 'Metastasis',
        'OS', 'OS_status']] # 'Microsatellite status', 'HER-2 status', 'DFS', 'DFS_status',
raw_df.rename(columns={'Histo Type': 'HistologicalType', 'Lauren Type': 'LaurenType',
                         'Pathological T stage': 'pT', 'Pathological N stage': 'pN', 'Metastasis': 'pM'}, inplace=True)
cox_base_df = prepare_multivariate_df(raw_df)

# ---------- 2 合并 数据 ----------
#读取文件summary_SXCH-Train_0.*.csv
file_path = glob.glob(f'{RESULT_DIR}/summary_SXCH-Train_0.*.csv')
result_df = pd.read_csv(file_path[0],dtype={'case_id': str})

cutoff_df = pd.read_csv(CUTOFF_PATH)
cutoff_all = cutoff_df.loc[cutoff_df['group'] == 'ALL', 'cutoff'].values[0]

final_df = merge_data(
        cox_base_df,
        result_df,
        cutoff_all
    )


'处理数据，让每个变量数值化：插补、映射——这个中心没有缺失的'
missing_counts = final_df.isnull().sum()
columns_to_impute = missing_counts[missing_counts > 0].index.tolist()
print('先查看缺失的列，再手工调整连续和分类变量:',columns_to_impute)

# categorical_cols = ['Tstage', 'Nstage', 'Mstage', 'TNMstage']
# continuous_cols = []

# # df_impute_missing_values = impute_missing_values(df, continuous_cols, categorical_cols)
# df_impute_missing_values = df.copy()


rename_dict = {
    'risk': f'GRASP',
    'risk_group': 'GRASP (H v L)',
    'Age_1': f"Age\n(>65 v \u226465)",
    'Gender_1': f'Gender\n(Male v Female)',
    'Location_2': f'Location\n(Body v Cardia)', 
    'Location_3': f'Location\n(Antrum v Cardia)', 
    'Location_4': f'Location\n(Whole v Cardia)', 
    'pT_2': f'pT\n(pT2 v pT1)', 
    'pT_3': f'pT\n(pT3 v pT1)', 
    'pT_4': f'pT\n(pT4 v pT1)', 
    'pN_1': f'pN\n(pN1 v pN0)', 
    'pN_2': f'pN\n(pN2 v pN0)', 
    'pN_3': f'pN\n(pN3 v pN0)', 
    'pM': f'pM\n(M1 vM0)', 
    'LaurenType_2': f'Lauren Type\n(Diffuse v Intesinal)', 
    'LaurenType_3': f'Lauren Type\n(Mixed v Intesinal)', 
    'Grade_2': f'Grade\n(Moderately v Well)',
    'Grade_3': f'Grade\n(Poorly v Well)',
    'HistologicalType_2': f'Histo. Type\n(Other v Adenocarcinoma)', 
    'Chemotherapy_1': f'Chemotherapy\n(Yes v No)', 
    'Stage_2': f'Stage (II v I)', 
    'Stage_3': f'Stage (III v I)', 
    'Stage_4': f'Stage (IV v I)', 
}

'单因素分析'
remove_cols = ['case_id', 'slide_id', 'slide_id_wax', 'survival_time', 'event_times', 'censorship', 'status', 'OS', 'OS_status']
varList = final_df.columns.tolist()
varList = [v for v in varList if v not in remove_cols]
uni_results_df = run_univariate_cox(final_df, varList)

save_univariate_cox_results(
    uni_results_df,
    save_path='PLOTS/3.uni_multi_cox/uni_cox.xlsx',
    rename_dict=rename_dict
)

'多因素分析'
rename_dict = {
    'risk': f'GRASP',
    # 'risk_group': 'Risk (H v L)',
    'Age_1': f"Age\n(>65 v \u226465)",
    # 'Gender_1': f'Gender\n(Male v Female)',
    'Location_2': f'Location\n(Body v Cardia)', 
    'Location_3': f'Location\n(Antrum v Cardia)', 
    'Location_4': f'Location\n(Whole v Cardia)', 
    'pT_2': f'pT\n(pT2 v pT1)', 
    'pT_3': f'pT\n(pT3 v pT1)', 
    'pT_4': f'pT\n(pT4 v pT1)', 
    'pN_1': f'pN\n(pN1 v pN0)', 
    'pN_2': f'pN\n(pN2 v pN0)', 
    'pN_3': f'pN\n(pN3 v pN0)', 
    'pM': f'pM\n(M1 vM0)', 
    'LaurenType_2': f'Lauren Type\n(Diffuse v Intesinal)', 
    'LaurenType_3': f'Lauren Type\n(Mixed v Intesinal)', 
    'Grade_2': f'Grade\n(Moderately v Well)',
    'Grade_3': f'Grade\n(Poorly v Well)',
    'HistologicalType_2': f'Histo. Type\n(Other v Adenocarcinoma)', 
    # 'Chemotherapy_1': f'Chemotherapy (Yes v No)', 
    # 'Stage_2': f'Stage (II v I)', 
    # 'Stage_3': f'Stage (III v I)', 
    # 'Stage_4': f'Stage (IV v I)', 
}
covariates = ['risk', 
              'pM', 'Age_1', 'pT_2', 'pT_3', 'pT_4', 'pN_1', 'pN_2', 'pN_3', #'Stage_2', 'Stage_3', 'Stage_4',
              'Grade_2', 'Grade_3', 'Location_2', 'Location_3', 'Location_4', 'HistologicalType_2', 'LaurenType_2', 'LaurenType_3', 
              ]
cph = run_multivariate_cox(final_df, covariates)

save_multivariate_cox_results(
    cph.summary,
    save_path='PLOTS/3.uni_multi_cox/multi_cox.xlsx',
    rename_dict=rename_dict
)

# plot_cox_forest(
#     cph,
#     rename_dict=rename_dict,
#     save_path='PLOTS/3.uni_multi_cox/multi_plot.svg',
#     fontsize=14
# )

plot_cox_forest_table(
    cph,
    rename_dict=rename_dict,
    save_path='PLOTS/3.uni_multi_cox/multi_plot.svg',
    fontsize=14
)
'合并单因素和多因素分析结果'
merge_uni_multi_cox(
    uni_path="PLOTS/3.uni_multi_cox/uni_cox.xlsx",
    multi_path="PLOTS/3.uni_multi_cox/multi_cox.xlsx",
    save_path="PLOTS/3.uni_multi_cox/uni&multi_cox.xlsx"
)
