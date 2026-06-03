import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test
import glob

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test
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

def new_p_value(high_risk, low_risk, data_df, group_type='risk_group'):
    """
    1 计算 Log-rank p 值
    2 使用 Cox 回归计算 HR 及置信区间
    3 返回可直接用于 KM 图标注的文本
    """

    logrank_res = logrank_test(
        high_risk['event_times'],
        low_risk['event_times'],
        event_observed_A=1 - high_risk['censorship'],
        event_observed_B=1 - low_risk['censorship']
    )

    test_df = pd.DataFrame({
        'status': 1 - data_df['censorship'],
        'survival_time': data_df['event_times'],
        group_type: data_df[group_type]
    })

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        df=test_df,
        duration_col='survival_time',
        event_col='status',
        formula=group_type
    )

    hr = cph.summary.loc[group_type, 'exp(coef)']
    hr_low = cph.summary.loc[group_type, 'exp(coef) lower 95%']
    hr_high = cph.summary.loc[group_type, 'exp(coef) upper 95%']

    text = (
        f"HR: {hr:.2f} (95% CI: {hr_low:.2f}-{hr_high:.2f})\n"
        f"Log-rank test {format_p_value(logrank_res.p_value)}"
    )

    return text

def plot_external_km(
    data_df,
    labels,
    plot_type,
    save_path,
    fontsiz=14,
    survival_state='OS',
    cutoff_time=None,
    set_xlim=False
):
    """
    1 根据 risk_group 绘制 KM 曲线
    2 支持截断时间
    3 自动添加 at-risk table
    4 自动计算 HR + log-rank 并标注
    """

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#1f77b4', '#d62728']
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
            color=colors[g]
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

        # kmf.fit(durations=data_df.loc[idx, 'survival_time'],event_observed=data_df.loc[idx, 'status'], label=labels[g])
        # kmf.plot_survival_function(ci_show=True, linewidth=3,show_censors=False,legend=False,color=colors[g])
        # kmfs.append(kmf)

    ax.set_ylim(0.05, 1.03)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel('Time (months)', fontsize=fontsiz)
    ax.set_ylabel(survival_state, fontsize=fontsiz)
    ax.tick_params(axis='both', labelsize=fontsiz)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if set_xlim:
        ax.set_xlim(-5, cutoff_time)
        xticks = np.arange(0, cutoff_time + 1, 20)
        ax.set_xticks(xticks)

    add_at_risk_counts(*kmfs, ax=ax, rows_to_show=['At risk'])
    fig = ax.get_figure()
    risk_ax = fig.axes[-1]
    risk_ax.tick_params(axis='x', labelsize=fontsiz)
    risk_ax.tick_params(axis='y', labelsize=fontsiz)


    group0 = data_df[data_df[plot_type] == 0]
    group1 = data_df[data_df[plot_type] == 1]

    stat_text = new_p_value(group1, group0, data_df, group_type=plot_type)

    ax.text(
        0.05, 0.05,
        stat_text,
        transform=ax.transAxes,
        fontsize=fontsiz,
        verticalalignment='bottom'
    )

    # ax.legend(
    # loc='upper right',
    # fontsize=fontsiz,
    # frameon=False
    # )

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

# =====================================================
# KM Plot
# =====================================================
def plot_internal_km(
    data_df,
    labels,
    plot_type,
    save_path,
    fontsiz=14,
    survival_state='OS',
    cutoff_time=None,
    center_name=None
):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))

    # colors = ['#1f77b4', '#d62728']
    colors = ['#d62728', '#1f77b4']
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
        sf = kmf.survival_function_#生存概率数据
        ci = kmf.confidence_interval_
        if cutoff_time is not None:
            sf = sf[sf.index <= cutoff_time]
            ci = ci[ci.index <= cutoff_time]

        ax.step(sf.index, sf[labels[g]], where='post', linewidth=3, color=colors[g], label=labels[g]) # where='post'表示事件发生后才下降
        ax.fill_between(  # 绘制置信区间的阴影区
            ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
            color=colors[g], alpha=0.15, step='post'
        )
        kmfs.append(kmf)

        # kmf.fit(durations=data_df.loc[idx, 'survival_time'],event_observed=data_df.loc[idx, 'status'].astype(int), label=labels[g])
        # kmf.plot_survival_function(ci_show=True, linewidth=3,show_censors=False,legend=False, color=colors[g])
        # kmfs.append(kmf)

    # ----- Axis Style -----
    ax.set_ylim(0.05, 1.03)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel('Time (months)', fontsize=fontsiz)
    ax.set_ylabel(survival_state, fontsize=fontsiz)
    ax.tick_params(axis='both', labelsize=fontsiz)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ----- At-risk Table -----
    add_at_risk_counts(*kmfs, ax=ax, rows_to_show=['At risk'])
    risk_ax = ax.get_figure().axes[-1]
    risk_ax.tick_params(axis='both', labelsize=fontsiz)

    ax.legend(loc='upper right', frameon=False, fontsize=fontsiz)
    if not center_name is None:
        ax.set_title(f'{center_name}', fontsize=fontsiz)

    # ----- Statistics -----
    group0 = data_df[data_df[plot_type] == 0]
    group1 = data_df[data_df[plot_type] == 1]

    stat_text = new_p_value(group1, group0, data_df, group_type=plot_type)

    ax.text(0.05, 0.05, stat_text, transform=ax.transAxes,
            fontsize=fontsiz, verticalalignment='bottom')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

CUTOFF_TIME = 120
FONTSIZE = 15

# RESULT_DIR = 'patient_level_RandomTrainData'
# RESULT_DIR = 'patient_level_seed42' 
RESULT_DIR = '0.0001loadloss'
SAVE_DIR = f'PLOTS/2.km/{RESULT_DIR}/化疗受益'
os.makedirs(SAVE_DIR, exist_ok=True)

centers = ['YYH', 'SYSUCC', 'JSPH', 'TCGA_STAD']
paths = glob.glob(f'results/moe_wsi/{RESULT_DIR}/results_external/summary_*_0.*.csv')

cutoff_df = pd.read_csv(f"PLOTS/2.km/{RESULT_DIR}/cutoff_media_risk_score.csv")


for center_name in centers:

    path = glob.glob(f'results/moe_wsi/{RESULT_DIR}/results_external/summary_{center_name}_0.*.csv')[0]
    print(path)
    if 'dfs' in path:
        survival_state = 'DFS'
    elif 'pfs' in path:
        survival_state = 'PFS'
    else:
        survival_state = 'OS'


    info_df = pd.read_csv(f'ori_files/{center_name}/clinical_info_all.csv',dtype={'case_id': str, 'slide_id': str})
    df = pd.read_csv(path,dtype={'case_id': str, 'slide_id': str})
    df = pd.merge(df, info_df, on='case_id', how='left')
    df['status'] = df[f'{survival_state}_status']
    df['event_times'] = df[survival_state]
    df.dropna(subset=['Chemotherapy'], inplace=True)

    agg_df = (
            df.groupby("case_id")
              .agg({
                  "risk": "mean",
              })
              .reset_index()
        )
    df = df.drop_duplicates(subset=['case_id']).drop(columns=['risk'])
    agg_df = pd.merge(agg_df, df, on='case_id', how='left')

    # ================= ALL 化疗受益 =================
    # 取group为ALL的cutoff_time
    cutoff_time = cutoff_df[cutoff_df['group'] == 'ALL']['cutoff'].values[0]

    #获得risk_group
    agg_df['risk_group'] = agg_df['risk'].apply(lambda x: 0 if x <= cutoff_time else 1)

    # set_xlim = True if 'YYH' in center_name else False
    # CUTOFF_TIME = 100 if 'YYH' in center_name else CUTOFF_TIME

    # set_xlim = True if 'STAD' in center_name else False
    # CUTOFF_TIME = 130 if 'STAD' in center_name else CUTOFF_TIME

    for risk_group in agg_df['risk_group'].unique():
        df_risk = agg_df[agg_df['risk_group'] == risk_group].copy()
        plot_internal_km(
                data_df=df_risk,
                labels=['No Chemo', 'Chemo'],
                plot_type='Chemotherapy',
                save_path=f'{SAVE_DIR}/Risk{risk_group}Chemo(ALL)_{center_name}.svg',
                fontsiz=FONTSIZE,
                cutoff_time=CUTOFF_TIME,
                survival_state=survival_state,
                center_name=center_name

            )

    # # ================= 亚组 化疗受益 =================
    subgroup_cutoffs = cutoff_df[cutoff_df['group'] != 'ALL']
    for _, row in subgroup_cutoffs.iterrows():

        group_name = row['group']
        subgroup_cutoff = row['cutoff']
        subgroup_col, subgroup_value = group_name.split('_', 1)
        if not subgroup_col in ['Stage']:
            continue
        if subgroup_value in ['1', '4']:
            continue
        
        # if not center_name == 'JSPH':
        #     continue

        #转成整数
        info_df[subgroup_col] = info_df[subgroup_col].apply(lambda x: round(x) if pd.notna(x) else x)
        info_df[subgroup_col] = info_df[subgroup_col].astype('Int64')
        subgroup_cases = info_df.loc[
            info_df[subgroup_col] == int(subgroup_value),
            'case_id'
        ]

        df_sub = agg_df[agg_df['case_id'].isin(subgroup_cases)].copy()
        df_sub['risk_group'] = (df_sub['risk'] > subgroup_cutoff).astype(int)


        for risk_group in df_sub['risk_group'].unique():
            df_risk = df_sub[df_sub['risk_group'] == risk_group].copy()
            
            if len(df_risk) < 10:
                print(f'Risk{risk_group}Chemo({group_name}) 样本太少，跳过')
                continue

            counts_dict_index = df_risk['Chemotherapy'].value_counts().sort_index().to_dict()
            if not 0 in counts_dict_index.keys() or not 1 in counts_dict_index.keys():
                print(f'{counts_dict_index} Risk{risk_group}Chemo({group_name}) 样本量不足，跳过')
                continue
            plot_internal_km(
                    data_df=df_risk,
                    labels=['No Chemo', 'Chemo'],
                    plot_type='Chemotherapy',
                    save_path=f'{SAVE_DIR}/Risk{risk_group}Chemo({group_name})_{center_name}.svg',
                    fontsiz=FONTSIZE,
                    cutoff_time=CUTOFF_TIME,
                    survival_state=survival_state,
                    center_name=center_name
                )



