import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test


# =====================================================
# Global Config
# =====================================================
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 15




# =====================================================
# Utility Functions
# =====================================================
def format_p_value(p_value, prefix=""):
    if p_value < 0.0001:
        return f"{prefix}p < 0.0001"
    return f"{prefix}p = {p_value:.4f}"


def new_p_value(high_risk, low_risk, data_df, group_type='risk_group'):
    """Calculate HR + log-rank statistics"""

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

    cph = CoxPHFitter()
    cph.fit(test_df, duration_col='survival_time', event_col='status', formula=group_type)

    hr = cph.summary.loc[group_type, 'exp(coef)']
    hr_low = cph.summary.loc[group_type, 'exp(coef) lower 95%']
    hr_high = cph.summary.loc[group_type, 'exp(coef) upper 95%']

    return (
        f"HR: {hr:.2f} (95% CI: {hr_low:.2f}-{hr_high:.2f})\n"
        f"Log-rank test {format_p_value(logrank_res.p_value)}"
    )


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

    # ----- Statistics -----
    group0 = data_df[data_df[plot_type] == 0]
    group1 = data_df[data_df[plot_type] == 1]

    stat_text = new_p_value(group1, group0, data_df, group_type=plot_type)

    ax.text(0.05, 0.05, stat_text, transform=ax.transAxes,
            fontsize=fontsiz, verticalalignment='bottom')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =====================================================
# Helper
# =====================================================
def find_center_name(path):
    sep1 = "summary_"
    sep2 = "_0."

    start_idx = path.find(sep1)
    end_idx = path.find(sep2)

    if start_idx == -1 or end_idx == -1:
        print("Center parse error:", path)
        return None

    result = path[start_idx + len(sep1): end_idx]

    if 'dfs' in result:
        return result.replace('_dfs', '-DFS')
    if 'pfs' in result:
        return result.replace('_pfs', '-PFS')

    return result


def determine_survival_type(path):
    if 'dfs' in path:
        return 'DFS'
    if 'pfs' in path:
        return 'PFS'
    return 'OS'


CUTOFF_TIME = 110
FONTSIZE = 15

RESULT_DIR = '0.0001loadloss'
SAVE_DIR = f'PLOTS/2.km/{RESULT_DIR}/化疗受益'
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# Data Load
# =====================================================
info_df = pd.read_csv(
    'ori_files/SXCH/clinical_info_all.csv',
    dtype={'case_id': str}
)

cutoff_df = pd.read_csv(f"PLOTS/2.km/{RESULT_DIR}/cutoff_media_risk_score.csv")
paths = glob.glob(f'results/moe_wsi/{RESULT_DIR}/summary_SXCH-Train_slide_0.*.csv') + \
        glob.glob(f'results/moe_wsi/{RESULT_DIR}/summary_SXCH-Val_slide_0.*.csv')


# =====================================================
# Main Loop
# =====================================================
for path in paths:
    # if not 'Val' in path:
    #     continue
    center_name = find_center_name(path)
    survival_state = determine_survival_type(path)

    df = pd.read_csv(path)
    df['status'] = 1 - df['censorship']
    df['event_times'] = df['survival_time']
    df = pd.merge(df, info_df[['case_id', 'Chemotherapy']], on='case_id', how='left')
    df.dropna(subset=['Chemotherapy'], inplace=True)
    if 'slide' in path:
        agg_df = (
            df.groupby("case_id")
              .agg({
                  "risk": "mean",
              })
              .reset_index()
        )
        df = df.drop_duplicates(subset=['case_id']).drop(columns=['risk'])
        df = pd.merge(agg_df, df, on='case_id', how='left')

    # ================= ALL 化疗受益 =================
    cutoff_all = cutoff_df.loc[cutoff_df['group'] == 'ALL', 'cutoff'].values[0]
    df_ALL = df.copy()
    df_ALL['risk_group'] = (df_ALL['risk'] > cutoff_all).astype(int)
    
    # print(df['Chemotherapy'].value_counts())
    for risk_group in df_ALL['risk_group'].unique():
        df_risk = df_ALL[df_ALL['risk_group'] == risk_group].copy()
        plot_internal_km(
                data_df=df_risk,
                labels=['No Chemo', 'Chemo'],
                plot_type='Chemotherapy',
                save_path=f'{SAVE_DIR}/Risk{risk_group}Chemo(ALL)_{center_name}.svg',
                fontsiz=FONTSIZE,
                cutoff_time=CUTOFF_TIME,
                survival_state=survival_state
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

        subgroup_cases = info_df.loc[
            info_df[subgroup_col].astype(str) == subgroup_value,
            'case_id'
        ]

        df_sub = df[df['case_id'].isin(subgroup_cases)].copy()
        df_sub['risk_group'] = (df_sub['risk'] > subgroup_cutoff).astype(int)

        for risk_group in df_sub['risk_group'].unique():
            df_risk = df_sub[df_sub['risk_group'] == risk_group].copy()
            
            if len(df_risk) < 10:
                print(f'Risk{risk_group}Chemo({group_name}) 样本太少，跳过')
                continue

            plot_internal_km(
                    data_df=df_risk,
                    labels=['No Chemo', 'Chemo'],
                    plot_type='Chemotherapy',
                    save_path=f'{SAVE_DIR}/Risk{risk_group}Chemo({group_name})_{center_name}.svg',
                    fontsiz=FONTSIZE,
                    cutoff_time=CUTOFF_TIME,
                    survival_state=survival_state
                )
