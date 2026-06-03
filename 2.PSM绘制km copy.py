import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test

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

CUTOFF_TIME = 110
FONTSIZE = 15

RESULT_DIR = '0.0001loadloss'
ROOT_DIR = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息'
SAVE_DIR = f'PLOTS/2.km/{RESULT_DIR}/化疗受益PSM'
cutoff_df = pd.read_csv(f"PLOTS/2.km/{RESULT_DIR}/cutoff_media_risk_score.csv")

for center in ['SXCH-Train', 'SXCH-Val', 'All_external']:
    info_df = pd.read_csv(f'{ROOT_DIR}/{center}_Info.csv', dtype={'case_id': str})
    survival_state = 'OS'
    info_df['censorship'] = 1 - info_df[survival_state+'_status']
    info_df['status'] = info_df[survival_state+'_status']
    info_df['event_times'] = info_df[survival_state]
    info_df['survival_time'] = info_df[survival_state]
    
    for sub_group in ['All', 'Stage_2', 'Stage_3']:
        df_cp = info_df.copy()
        if sub_group == 'All':
            cutoff_ = cutoff_df.loc[cutoff_df['group'] == 'ALL', 'cutoff'].values[0]
        else:
            subgroup_col, subgroup_value = sub_group.split('_', 1)
            subgroup_cases = df_cp.loc[
            df_cp[subgroup_col].astype(str) == subgroup_value,
            'case_id'
            ]

            df_cp = df_cp[df_cp['case_id'].isin(subgroup_cases)].copy()
            cutoff_ = cutoff_df.loc[cutoff_df['group'] == sub_group, 'cutoff'].values[0]

        print(f'处理{center} ({sub_group})')
        file_path = f'{ROOT_DIR}/PSM_{sub_group}.xlsx'
        psm_df = pd.read_excel(file_path, sheet_name=center)
        case_id = psm_df['case_id'].tolist()
        km_df = df_cp[df_cp['case_id'].isin(case_id)]
        km_df['risk_group'] = (km_df['risk'] > cutoff_).astype(int)
        print(km_df.columns)

        for risk_group in km_df['risk_group'].unique():
            df_risk = km_df[km_df['risk_group'] == risk_group].copy()
            plot_internal_km(
                    data_df=df_risk,
                    labels=['No Chemo', 'Chemo'],
                    plot_type='Chemotherapy',
                    save_path=f'{SAVE_DIR}/Risk{risk_group}Chemo(ALL)_{center}_{sub_group}.svg',
                    fontsiz=FONTSIZE,
                    cutoff_time=CUTOFF_TIME,
                    survival_state=survival_state
                )
            
        # fdsfd