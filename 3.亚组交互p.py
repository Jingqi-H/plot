'训练集做单多因素分析，得到具有显著性的因素'

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from matplotlib.gridspec import GridSpec
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from scipy.stats import chi2
import os



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
        "P value": "Uni_P",
        "P": "Uni_P (val)",
    })

    multi_df = multi_df.rename(columns={
        "HR": "Multi_HR",
        "95% CI": "Multi_CI",
        "P value": "Multi_P",
        "P": "Multi_P (val)",
    })

    # =========================
    # 3️⃣ merge
    # =========================
    merged = pd.merge(
        uni_df[["Covariate", "Uni_HR", "Uni_CI", "Uni_P", "Uni_P (val)"]],
        multi_df[["Covariate", "Multi_HR", "Multi_CI", "Multi_P", "Multi_P (val)"]],
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
        "P (Uni) (val)": merged["Uni_P (val)"],
        "P (Uni)": merged["Uni_P"],
        "Multivariate HR (95% CI)": merged["Multivariate HR (95% CI)"],
        "P (Multi) (val)": merged["Multi_P (val)"],
        "P (Multi)": merged["Multi_P"],
    })

    # =========================
    # 6️⃣ 处理多因素空值
    # =========================
    df_final["Multivariate HR (95% CI)"] = df_final["Multivariate HR (95% CI)"].fillna("—")
    df_final["P (Multi)"] = df_final["P (Multi)"].fillna("—")
    df_final["P (Multi) (val)"] = df_final["P (Multi) (val)"].fillna("—")

    # =========================
    # 7️⃣ 保存
    # =========================
    df_final.to_excel(save_path, index=False)

    print("✅ 合并完成：", save_path)

    return df_final

# ============================================================
#  构建 Cox 用多因素数据
# ============================================================
def prepare_multivariate_df(df, exclude_cols=['risk','case_id', 'OS', 'DFS', 'survival_time','event_times','Microsatellite status','HER-2 status','center']):
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


    # cox_df = pd.get_dummies(
    #     cox_df,
    #     columns=['Age', 'pT', 'pN',  'Gender', 'Location',  'HistologicalType',
    #    'LaurenType', 'Chemotherapy', 'Stage', 'Grade'],
    #     drop_first=True
    # )

    return cox_df

# ============================================================
#  合并 Cox 临床数据 + k 折预测结果
# ============================================================
def merge_data(
    
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

    final_df = result_df.copy()

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

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()





# def save_univariate_cox_results(
#     uni_results_df,
#     save_path,
#     rename_dict=None,
# ):
#     """
#     保存单因素 Cox 结果到 Excel
#     Sheet1: 原始结果
#     Sheet2: 论文格式（Variable + HR + 95% CI + P value）
#     """

#     # =====================
#     # Sheet 1：原始结果
#     # =====================
#     raw_summary = uni_results_df.copy()
#     raw_summary.index.name = 'index'

#     # =====================
#     # Sheet 2：论文格式整理
#     # =====================
#     summary = raw_summary.copy()

#     summary['HR'] = summary['exp(coef)']
#     summary['HR_lower'] = summary['exp(coef) lower 95%']
#     summary['HR_upper'] = summary['exp(coef) upper 95%']

#     paper_df = pd.DataFrame({
#         'Variable': summary['Variable'],
#         'HR': summary['HR'].round(4),
#         '95% CI': summary['HR_lower'].round(2).astype(str)
#                   + '–'
#                   + summary['HR_upper'].round(2).astype(str),
#         'P': summary['p']
#     })

#     # 变量重命名（可选）
#     if rename_dict is not None:
#         paper_df['Variable'] = paper_df['Variable'].map(
#             lambda x: rename_dict.get(x, x)
#         )

#     # P值格式
#     paper_df['P value'] = paper_df['P'].apply(
#         lambda x: f'{x:.2g}' if x >= 0.05 else '<0.05'
#     )

#     # =====================
#     # 写 Excel
#     # =====================
#     with pd.ExcelWriter(save_path) as writer:
#         raw_summary.to_excel(writer, sheet_name='Raw_Univariate_Cox')
#         paper_df.to_excel(writer, sheet_name='Univariate_Cox', index=False)

#     print(f"单因素Cox结果已保存至: {save_path}")

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

def format_p(p):
    if p < 1e-3:
        return f"{p:.2e}"   # 科学计数法
    else:
        return f"{p:.2f}"
    
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
        'P': summary['p']
    })

    if rename_dict is not None:

        # 重命名
        paper_df['Covariate'] = paper_df['Covariate'].map(
            lambda x: rename_dict.get(x, x)
        )

        # 🔥 强制顺序按 rename_dict
        ordered_vars = [rename_dict[k] for k in rename_dict if k in summary['Covariate'].values]
        paper_df = paper_df.set_index('Covariate').loc[ordered_vars].reset_index()

    paper_df['P value'] = paper_df['P'].apply(
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
        'P': summary['p']
    })

    if rename_dict is not None:

        paper_df['Covariate'] = paper_df['Covariate'].map(
            lambda x: rename_dict.get(x, x)
        )

        ordered_vars = [rename_dict[k] for k in rename_dict if k in summary.index]
        paper_df = paper_df.set_index('Covariate').loc[ordered_vars].reset_index()

    paper_df['P value'] = paper_df['P'].apply(
        lambda x: f'{x:.2g}' if x >= 0.05 else '<0.05'
    )

    with pd.ExcelWriter(save_path) as writer:
        raw_summary.to_excel(writer, sheet_name='Raw_Cox_Summary')
        paper_df.to_excel(writer, sheet_name='Multivariate_Cox', index=False)


def feats_stardand(df, featsName=None): 
    df[featsName] = ( df[featsName] + 0.51 ).astype(int) 
    return df 


def impute_missing_values(
    df,
    columns_to_impute,
    random_state=0,
    max_iter=10
):
    continuous_cols= columns_to_impute.copy()

    df_copy = df.copy(deep=True)
    cont_subset = df_copy[continuous_cols].apply(
        pd.to_numeric, errors='coerce'
    )
    valid_cont_cols = cont_subset.columns[cont_subset.notna().any()]

    if len(valid_cont_cols) > 0:
        cont_imputer = IterativeImputer(
            random_state=random_state,
            max_iter=max_iter
        )
        cont_imputed = cont_imputer.fit_transform(
            cont_subset[valid_cont_cols]
        )

        df_copy[valid_cont_cols] = pd.DataFrame(
            cont_imputed,
            columns=valid_cont_cols,
            index=df_copy.index
        )
        df_copy = feats_stardand(df_copy, valid_cont_cols)
    df_copy['Lauren Type'] = df_copy['Lauren Type'].map({0:1,1:1,2:2,3:3})
    df_copy['Stage'] = df_copy['Stage'].map({0:1,1:1,2:2,3:3,4:4,5:4})

    return df_copy

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from scipy.stats import chi2


def format_p(p):
    if pd.isna(p):
        return ""
    elif p < 0.001:
        return "<0.001"
    else:
        return f"{p:.3f}"


def format_event_count(sub_df, risk_col, event_col, risk_value):
    """
    生成：事件人数/总人数(百分比)
    """
    group_df = sub_df[sub_df[risk_col] == risk_value]
    total = len(group_df)
    events = group_df[event_col].sum()

    if total == 0:
        return "0/0 (0.0%)"

    percent = events / total * 100
    return f"{int(events)}/{int(total)} ({percent:.1f}%)"


# def compute_interaction_p(df, time_col, event_col, risk_col, var, cal_method='lrt'):

#     if cal_method == 'lrt':
#         cph1 = CoxPHFitter(penalizer=0.1)
#         cph1.fit(df[[time_col, event_col, risk_col, var]],
#                     duration_col=time_col,
#                     event_col=event_col)

#         df_inter = df.copy()
#         df_inter['interaction'] = df[risk_col] * df[var]

#         cph2 = CoxPHFitter(penalizer=0.1)
#         cph2.fit(df_inter[[time_col, event_col, risk_col, var, 'interaction']],
#                     duration_col=time_col,
#                     event_col=event_col)

#         LR_stat = 2 * (cph2.log_likelihood_ - cph1.log_likelihood_)
#         p_interaction = chi2.sf(LR_stat, df=1)
#     elif cal_method == 'wald_test':
#         pass

#     else:
#         raise ValueError(f'cal_method={cal_method} 不支持的 method.')

#     return p_interaction

def compute_interaction_p(df, time_col, event_col, risk_col, var, cal_method='lrt'):

    try:
        if cal_method == 'lrt':
            # ===============================
            # LRT 方法
            # ===============================
            cph1 = CoxPHFitter(penalizer=0.1)
            cph1.fit(df[[time_col, event_col, risk_col, var]],
                     duration_col=time_col,
                     event_col=event_col)

            df_inter = df.copy()
            df_inter['interaction'] = df[risk_col] * df[var]

            cph2 = CoxPHFitter(penalizer=0.1)
            cph2.fit(df_inter[[time_col, event_col, risk_col, var, 'interaction']],
                     duration_col=time_col,
                     event_col=event_col)

            LR_stat = 2 * (cph2.log_likelihood_ - cph1.log_likelihood_)
            p_interaction = chi2.sf(LR_stat, df=1)

        elif cal_method == 'wald_test':
            # ===============================
            # Wald test 方法（推荐用 formula）
            # ===============================
            cph = CoxPHFitter(penalizer=0.1)

            # formula = f"{risk_col} + {var} + {risk_col}:{var}"
            formula = f"{risk_col} + `{var}` + {risk_col}:`{var}`"
            cph.fit(df,
                    duration_col=time_col,
                    event_col=event_col,
                    formula=formula)

            interaction_term = f"{risk_col}:{var}"

            # 有些情况下可能名称会有变化（极少数）
            if interaction_term in cph.summary.index:
                p_interaction = cph.summary.loc[interaction_term, 'p']
            else:
                # fallback（更鲁棒）
                p_interaction = np.nan

        else:
            raise ValueError(f'cal_method={cal_method} 不支持的 method.')

    except Exception as e:
        print(f"[Warning] interaction p 计算失败: {var}, error={e}")
        p_interaction = np.nan

    return p_interaction


def subgroup_analysis(
    df,
    cutoff_df,
    time_col='survival_time',
    event_col='status',
    risk_col='risk_group',
    subgroup_vars=None,
    template_path="PLOTS/@source/template_subgroup.xlsx",
    output_path="subgroup_analysis.xlsx"
):
    """
    主函数：亚组分析 + interaction p + 模板替换
    """

    if subgroup_vars is None:
        subgroup_vars = ['Age', 'Gender', 'CEA', 'CA199', 'Location', 'Grade',
                         'Histological Type', 'Lauren Type', 'Chemotherapy',
                         'Stage', 'pT', 'pN', 'pM']

    results = []

    for var in subgroup_vars:

        levels = sorted(df[var].dropna().unique())
        df.dropna(subset=[var], inplace=True)

        # 计算 interaction p（整个变量一个）
        p_interaction = compute_interaction_p(df, time_col, event_col, risk_col, var)

        for lvl in levels:
            print(f'var={var}, lvl={lvl}')

            cutoff = cutoff_df[cutoff_df['group'] == f'{var}_{str(lvl)}']['cutoff'].values[0]

            sub_df = df[df[var] == lvl].copy()
            sub_df['risk_group'] = (sub_df['risk'] > cutoff).astype(int)

            # 必须同时有高低风险
            if sub_df[risk_col].nunique() < 2:
                continue
            
            # 把 可空整数Int64 → 标准整数int64，同时处理空值（NaN填0，生存分析删失）
            sub_df[event_col] = sub_df[event_col].fillna(0).astype('int64')
            cph = CoxPHFitter(penalizer=0.1) # penalizer=0.1
            cph.fit(sub_df[[time_col, event_col, risk_col]],
                    duration_col=time_col,
                    event_col=event_col)

            summary = cph.summary.loc[risk_col]
            hr = np.exp(summary['coef'])
            ci_lower = np.exp(summary['coef lower 95%'])
            ci_upper = np.exp(summary['coef upper 95%'])
            p_value = summary['p']

            # Low / High risk 事件统计
            low_str = format_event_count(sub_df, risk_col, event_col, 0)
            high_str = format_event_count(sub_df, risk_col, event_col, 1)

            results.append({
                'Subgroup': f"{var}={lvl}",
                'Low Risk': low_str,
                'High Risk': high_str,
                'HR(95%CI)': f"{hr:.2f} ({ci_lower:.2f}-{ci_upper:.2f})",
                'P (val)': p_value,
                'P': format_p(p_value),
                'P_interaction (val)': p_interaction,
                'P_interaction': format_p(p_interaction)
            })

            # except:
            #     continue

    result_df = pd.DataFrame(results)

    # ===============================
    # 模板替换 Subgroup 名称
    # ===============================
    template_df = pd.read_excel(template_path)

    mapping = dict(zip(template_df['Subgroup'], template_df['new']))

    result_df['Subgroup'] = result_df['Subgroup'].map(mapping).fillna(result_df['Subgroup'])

    # ===============================
    # 保存
    # ===============================
    result_df.to_excel(output_path, index=False)

    print(f"✅ 结果已保存到: {output_path}")

    return result_df



# exp_id = '0.0001loadloss'
exp_id = 'final'

RESULT_DIR = f"results/moe_wsi/{exp_id}"
INFO_PATH = "ori_files/SXCH/clinical_info_all.csv"
CUTOFF_PATH = f"PLOTS/2.km/{exp_id}/cutoff_media_risk_score.csv"
SAVE_DIR = f"PLOTS/3.uni_multi_cox/亚组交互p"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------- 2 合并 数据 ----------
# center = 'SXCH-Train'
# center = 'SXCH-Val'
# center = 'YYH'
# center = 'SYSUCC'
# center = 'External'
# center = 'TCGA_STAD'
for center in ['SXCH-Train', 'SXCH-Val', 'CMU1H', 'YYH', 'SYSUCC', 'TCGA_STAD', 'External']:
    print(f'========================== {center}')
    file_path = f'PLOTS/2.km/{exp_id}/临床&Risk信息/{center}_Info.csv'
    result_df = pd.read_csv(file_path,dtype={'case_id': str})

    cc = ['case_id', 'risk', 'Age', 'Gender', 'CEA', 'CA199', 'Location', 'Grade', 'Histo Type',
        'Lauren Type', 'Chemotherapy',
        'Stage', 'Pathological T stage', 'Pathological N stage', 'Metastasis',
        'DFS', 'DFS_status', 'OS', 'OS_status', 'center']
    if center == 'TCGA_STAD':
        cc.remove('CEA')
        cc.remove('CA199')
    result_df = result_df[cc]

    result_df['survival_time'] = result_df['OS']
    result_df['censorship'] = 1- result_df['OS_status']

    # result_df.rename(columns={'Histo Type': 'HistologicalType', 'Lauren Type': 'LaurenType',
    #                          'Pathological T stage': 'pT', 'Pathological N stage': 'pN', 'Metastasis': 'pM'}, inplace=True)

    cutoff_df = pd.read_csv(CUTOFF_PATH)
    cutoff_all = cutoff_df.loc[cutoff_df['group'] == 'ALL', 'cutoff'].values[0]
    final_df = merge_data(result_df, cutoff_all)


    '处理数据，让每个变量数值化：插补、映射——这个中心没有缺失的'
    missing_counts = final_df.isnull().sum()
    columns_to_impute = missing_counts[missing_counts > 0].index.tolist()
    remove_cols = ['DFS', 'DFS_status', 'OS', 'OS_status',
                'Microsatellite status', 'HER-2 status', 'PFS', 'PFS_status']
    columns_to_impute = [col for col in columns_to_impute if col not in remove_cols]
    print('缺失的列:', columns_to_impute)
    final_df = impute_missing_values(final_df, columns_to_impute)
    final_df = prepare_multivariate_df(final_df)

    subgroup_vars = ['Age', 'Gender', 'CEA', 'CA199', 'Location', 'Grade', 'Histo Type', 'Lauren Type', 
                        'Chemotherapy', 'Stage', 'Pathological T stage', 'Pathological N stage', 
                        'Metastasis']
    if center == 'TCGA_STAD':
            subgroup_vars.remove('CEA')
            subgroup_vars.remove('CA199')

    result_df = subgroup_analysis(
        final_df,
        cutoff_df,
        subgroup_vars = subgroup_vars,
        template_path="PLOTS/@source/template_subgroup.xlsx",
        output_path=f"{SAVE_DIR}/subgroup_analysis_{center}.xlsx"
    )
