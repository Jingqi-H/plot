import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
import glob
from lifelines.statistics import logrank_test


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 15





def format_p_value(p_value, prefix=""):
    """
    统一格式化 p 值显示
    """
    if p_value < 0.0001:
        return f"{prefix}p < 0.0001"
    else:
        return f"{prefix}p = {p_value:.4f}"


def new_text(data_df, group_type):
    test_df_cox = pd.DataFrame({
        'status': data_df['status'],  # status = 1 - censorship
        'survival_time': data_df['survival_time'],  # survival_time = event_times
        'risk_group': data_df[group_type]  # risk_group ????
    })
    test_df_cox = pd.get_dummies(test_df_cox, columns=['risk_group'], drop_first=True)

    cph = CoxPHFitter()
    cph.fit(df=test_df_cox,duration_col='survival_time',event_col='status')
    summary = cph.summary

    text_lines = []
    for i, row in summary.iterrows():
        if i == 'risk_group_1':
            text = 'Medium risk'
        elif i == 'risk_group_2':
            text = 'High risk'
        hr = row['exp(coef)']
        ci_lower = row['exp(coef) lower 95%']
        ci_upper = row['exp(coef) upper 95%']
        p = row['p']
        text_lines.append(f"{text}: HR {hr:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}), {format_p_value(p)}")


    logrank_result = logrank_test(data_df['survival_time'], data_df['risk_group'], data_df['status'])
    text_lines.append(f"Log-rank test {format_p_value(logrank_result.p_value)}")
    return text_lines

# =====================================================
# External KM Plot for Tertile Groups
# =====================================================
def plot_external_tertile_km(
    data_df,
    labels,
    plot_type,
    save_path,
    fontsiz=14,
    survival_state='OS',
    cutoff_time=None,
    set_xlim=False,
    legend=True,
):
    """
    1 根据 risk_group 绘制三分位 KM 曲线
    2 支持截断时间
    3 自动添加 at-risk table
    4 自动计算 HR + log-rank 并标注
    """

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#1f77b4', '#ff7f0e', '#d62728']  # 蓝、橙、红
    groups = np.sort(data_df[plot_type].astype(int).unique())

    kmfs = []

    for g in groups:
        kmf = KaplanMeierFitter()
        idx = data_df[plot_type] == g

        kmf.fit(
            durations=data_df.loc[idx, 'survival_time'],
            event_observed=data_df.loc[idx, 'status'],
            label=labels[g]
        )

        sf = kmf.survival_function_
        ci = kmf.confidence_interval_

        if cutoff_time is not None:
            sf = sf[sf.index <= cutoff_time]
            ci = ci[ci.index <= cutoff_time]

        ax.step(
            sf.index,
            sf[labels[g]],
            where='post',
            linewidth=3,
            color=colors[g],
            label=labels[g]
        )

        ax.fill_between(
            ci.index,
            ci.iloc[:, 0],
            ci.iloc[:, 1],
            color=colors[g],
            alpha=0.15,
            step='post'
        )

        kmfs.append(kmf)

    # ----- Legend -----
    if legend:
        ax.legend(loc='upper right', frameon=False, fontsize=fontsiz)
    
    # ----- Axis Style -----
    ax.set_ylim(0.05, 1.03)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel('Time (months)', fontsize=fontsiz)
    ax.set_ylabel(survival_state, fontsize=fontsiz)
    ax.tick_params(axis='both', labelsize=fontsiz)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if set_xlim:
        ax.set_xlim(-5, cutoff_time-30)
        xticks = np.arange(0, cutoff_time + 1, 20)
        ax.set_xticks(xticks)

    add_at_risk_counts(*kmfs, ax=ax, rows_to_show=['At risk'])
    fig = ax.get_figure()
    risk_ax = fig.axes[-1]
    risk_ax.tick_params(axis='x', labelsize=fontsiz)
    risk_ax.tick_params(axis='y', labelsize=fontsiz)

    # ----- Statistics -----
    # 对于三分位法，我们比较低风险组和高风险组
    group_low = data_df[data_df[plot_type] == 0]
    group_high = data_df[data_df[plot_type] == 2]

    if len(group_low) > 0 and len(group_high) > 0:
        # stat_text = new_p_value(group_high, group_low, data_df, group_type=plot_type)
        # ax.text(
        #     0.05, 0.05,
        #     stat_text,
        #     transform=ax.transAxes,
        #     fontsize=fontsiz,
        #     verticalalignment='bottom'
        # )
        combined_text = new_text(data_df, group_type=plot_type)
        ax.text(0.05, 0.05, '\n'.join(combined_text), transform=ax.transAxes,
                fontsize=fontsiz, verticalalignment='bottom')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def find_center_name(s):
    sep1 = "summary_"
    sep2 = "_0."

    start_idx = s.find(sep1)
    end_idx = s.find(sep2)

    # 提取中间字符（含异常处理，避免分隔符不存在报错）
    if start_idx != -1 and end_idx != -1 and start_idx + len(sep1) < end_idx:
        result = s[start_idx + len(sep1) : end_idx]
        if 'dfs' in result:
            result = result.replace('_dfs','-DFS')
        elif 'pfs' in result:
            result = result.replace('_pfs','-PFS')
        else:
            pass
        # print("提取结果：", result)  # 输出：SXCH_Val
        return result

    else:
        print("错误：字符串中未找到指定分隔符，或分隔符顺序异常",s)
        return None


CUTOFF_TIME = 110
FONTSIZE = 15

RESULT_DIR = '0.0001loadloss'

paths = glob.glob(f'results/moe_wsi/{RESULT_DIR}/results_external/summary_*.csv')

cutoff_df = pd.read_csv(f"PLOTS/2.km/{RESULT_DIR}_tertile/cutoff_tertile_risk_score.csv")

labels=('Low Risk', 'Intermediate Risk', 'High Risk')

for path in paths:
    
    center_name = find_center_name(path)
    if 'dfs' in path:
        survival_state = 'DFS'
    elif 'pfs' in path:
        survival_state = 'PFS'
    else:
        survival_state = 'OS'

    save_path = f'PLOTS/2.km/{RESULT_DIR}_tertile/ALL_{center_name}.svg'

    df = pd.read_csv(path)

    agg_df = (
            df.groupby("case_id")
              .agg({
                  "risk": "mean",
                  "survival_time": "first",
                  "censorship": "first"
              })
              .reset_index()
        )

    # 取group为ALL的cutoff值
    cutoff_all = cutoff_df[cutoff_df['group'] == 'ALL']
    cutoff1 = cutoff_all['cutoff1'].values[0]
    cutoff2 = cutoff_all['cutoff2'].values[0]

    # 划分三个风险组：0=低风险, 1=中风险, 2=高风险
    agg_df['risk_group'] = 0  # 默认低风险
    agg_df.loc[agg_df['risk'] > cutoff1, 'risk_group'] = 1  # 中风险
    agg_df.loc[agg_df['risk'] > cutoff2, 'risk_group'] = 2  # 高风险
    
    agg_df['status'] = 1 - agg_df['censorship']
    agg_df['event_times'] = agg_df['survival_time']

    set_xlim = True if 'STAD' in center_name else False
    CUTOFF_TIME = 120 if 'STAD' in center_name else CUTOFF_TIME
    plot_external_tertile_km(
        data_df=agg_df,
        labels=list(labels),
        plot_type='risk_group',
        save_path=save_path,
        fontsiz=FONTSIZE,
        cutoff_time=CUTOFF_TIME,
        survival_state=survival_state,
        set_xlim=set_xlim,
        legend=False,
    )