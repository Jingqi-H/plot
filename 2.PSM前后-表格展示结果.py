import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from scipy.stats import chi2

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

    cph = CoxPHFitter(penalizer=0.1)
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
    title=None
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
    if title is not None:
        ax.set_title(title, fontsize=fontsiz)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def filter_psm_df(psm_df,df_cp):
    # 【核心修改】筛选【同时存在】的配对ID
    valid_case_ids = []
    # 遍历每一对匹配样本
    for _, row in psm_df.iterrows():
        treated_id = row['Treated_CaseID']
        control_id = row['Control_CaseID']
        
        # 关键判断：两个ID必须【同时存在】于df_cp中
        if (treated_id in df_cp['case_id'].values) and (control_id in df_cp['case_id'].values):
            valid_case_ids.append(treated_id)
            valid_case_ids.append(control_id)

    # 用【有效配对ID】筛选最终KM分析数据集
    km_df = df_cp[df_cp['case_id'].isin(valid_case_ids)].copy()
    return km_df


def prepare_survival(df, endpoint='OS'):
    df = df.copy()
    df['censorship'] = 1 - df[f'{endpoint}_status']
    df['status'] = df[f'{endpoint}_status']
    df['event_times'] = df[endpoint]
    df['survival_time'] = df[endpoint]
    return df


def get_subgroup_df(df, sub_group, cutoff_df):
    df_cp = df.copy()

    if sub_group == 'All':
        cutoff_ = cutoff_df.loc[cutoff_df['group'] == 'ALL', 'cutoff'].values[0]
    else:
        subgroup_col, subgroup_value = sub_group.split('_', 1)

        df_cp[subgroup_col] = df_cp[subgroup_col].apply(lambda x: round(x) if pd.notna(x) else x)
        df_cp[subgroup_col] = df_cp[subgroup_col].astype('Int64')

        df_cp = df_cp[df_cp[subgroup_col] == int(subgroup_value)].copy()

        cutoff_ = cutoff_df.loc[cutoff_df['group'] == sub_group, 'cutoff'].values[0]

    df_cp['risk_group'] = (df_cp['risk'] > cutoff_).astype(int)

    return df_cp


def compute_cox(df):
    df = df.copy()

    # ----------------------
    # ⚠️ 基础检查
    # ----------------------
    if df['Chemotherapy'].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan

    if df['status'].sum() == 0:
        return np.nan, np.nan, np.nan, np.nan

    # ----------------------
    # ⚠️ 编码（确保是0/1）
    # ----------------------
    df['Chemotherapy'] = df['Chemotherapy'].astype(int)

    # ----------------------
    # 📊 Cox模型
    # ----------------------
    cph = CoxPHFitter(penalizer=0.1)

    try:
        cph.fit(
            df[['survival_time', 'status', 'Chemotherapy']],
            duration_col='survival_time',
            event_col='status'
        )

        summary = cph.summary.loc['Chemotherapy']

        hr = summary['exp(coef)']
        ci_low = summary['exp(coef) lower 95%']
        ci_high = summary['exp(coef) upper 95%']
        p = summary['p']

        return hr, ci_low, ci_high, p

    except:
        return np.nan, np.nan, np.nan, np.nan


def compute_interaction_p(df, cal_method='lrt'):
    df = df.copy()

    # ----------------------
    # ⚠️ 基础检查
    # ----------------------
    if df['Chemotherapy'].nunique() < 2:
        return np.nan

    if df['risk_group'].nunique() < 2:
        return np.nan

    if df['status'].sum() == 0:
        return np.nan

    # ----------------------
    # ⚠️ 编码
    # ----------------------
    df['Chemotherapy'] = df['Chemotherapy'].astype(int)
    df['risk_group'] = df['risk_group'].astype(int)

    # ----------------------
    # 📊 Cox with interaction
    # ----------------------
    if cal_method == 'lrt':
        cph1 = CoxPHFitter(penalizer=0.1)
        cph1.fit(df[['survival_time', 'status', 'Chemotherapy', 'risk_group']],
                    duration_col='survival_time',
                    event_col='status')

        df_inter = df.copy()
        df_inter['interaction'] = df['Chemotherapy'] * df['risk_group']

        cph2 = CoxPHFitter(penalizer=0.1)
        cph2.fit(df_inter[['survival_time', 'status', 'Chemotherapy', 'risk_group', 'interaction']],
                    duration_col='survival_time',
                    event_col='status')

        LR_stat = 2 * (cph2.log_likelihood_ - cph1.log_likelihood_)
        p_interaction = chi2.sf(LR_stat, df=1)
        return p_interaction
    elif cal_method == 'wald_test':    
        cph = CoxPHFitter(penalizer=0.1)

        try:
            cph.fit(
                df[['survival_time', 'status', 'Chemotherapy', 'risk_group']],
                duration_col='survival_time',
                event_col='status',
                formula="Chemotherapy * risk_group"
            )

            summary = cph.summary

            # interaction项名称通常是：
            # Chemotherapy:risk_group
            interaction_term = 'Chemotherapy:risk_group'

            if interaction_term in summary.index:
                p_interaction = summary.loc[interaction_term, 'p']
            else:
                p_interaction = np.nan

            return p_interaction

        except:
            return np.nan
    


CUTOFF_TIME = 110
FONTSIZE = 15

RESULT_DIR = 'final'

ROOT_DIR = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息'
SAVE_DIR = f'PLOTS/2.km/{RESULT_DIR}/化疗受益表格'
os.makedirs(SAVE_DIR, exist_ok=True)

cutoff_df = pd.read_csv(f"PLOTS/2.km/{RESULT_DIR}/cutoff_media_risk_score.csv")

# ==============================
# 🔧 工具函数（强烈建议这样拆）
# ==============================

# ==============================
# 🌟 主流程
# ==============================

before_results = []
after_results = []

for center in ['SXCH-Train', 'SXCH-Val', 'External']:


    print(f'\n========== {center} ==========')

    psm_path = f'{ROOT_DIR}/PSM_{center}.xlsx'

    # ----------------------
    # 📥 读取数据
    # ----------------------
    if center in ['CMU1H','YYH', 'SYSUCC', 'TCGA_STAD']:
        info_df = pd.read_csv(f'{ROOT_DIR}/External_Info.csv', dtype={'case_id': str})
        info_df = info_df[info_df['center'] == center]
    else:
        info_df = pd.read_csv(f'{ROOT_DIR}/{center}_Info.csv', dtype={'case_id': str})

    # 基础筛选
    info_df = info_df[(info_df['Stage'] != 1) & (info_df['Stage'] != 4)]
    info_df = info_df[info_df['center'] != 'JSPH']
    info_df.dropna(subset=['Chemotherapy'],inplace=True)

    info_df = prepare_survival(info_df, endpoint='OS')

    # ----------------------
    # 🔁 subgroup循环
    # ----------------------
    for sub_group in ['All', 'Stage_2', 'Stage_3']:

        df_cp = get_subgroup_df(info_df, sub_group, cutoff_df)

        # ----------------------
        # ⭐ interaction（PSM前）
        # ----------------------
        p_inter_before = compute_interaction_p(df_cp)

        # ----------------------
        # ⭐ interaction（PSM后）
        # ⚠️ 需要合并所有risk_group
        # ----------------------
        after_psm_all_list = []

        for risk_group in df_cp['risk_group'].unique():

            psm_df = pd.read_excel(
                psm_path,
                sheet_name=f'{sub_group}#{risk_group}',
                dtype={'Treated_CaseID': str, 'Control_CaseID': str}
            )

            before_psm = df_cp[df_cp['risk_group'] == risk_group].copy()
            after_psm = filter_psm_df(psm_df, before_psm)

            after_psm_all_list.append(after_psm)

        after_psm_all = pd.concat(after_psm_all_list, axis=0)

        p_inter_after = compute_interaction_p(after_psm_all)

        # ----------------------
        # 🔁 risk_group循环
        # ----------------------
        for risk_group in df_cp['risk_group'].unique():

            psm_df = pd.read_excel(
                psm_path,
                sheet_name=f'{sub_group}#{risk_group}',
                dtype={'Treated_CaseID': str, 'Control_CaseID': str}
            )

            before_psm = df_cp[df_cp['risk_group'] == risk_group].copy()
            after_psm = filter_psm_df(psm_df, before_psm)

            print(f'{center} | {sub_group} | risk{risk_group} | '
                  f'PSM前: {len(before_psm)} → PSM后: {len(after_psm)}')

            # ----------------------
            # 📊 Cox（PSM前）
            # ----------------------
            hr, ci_l, ci_h, p = compute_cox(before_psm)

            before_results.append({
                'cohort': center,
                'stage': sub_group,
                'risk_group': risk_group,
                'No Chemo/Chemo(n)': f'{len(before_psm[before_psm["Chemotherapy"]==0])}/{len(before_psm[before_psm["Chemotherapy"]==1])}',
                'HR(95%CI)': f"{hr:.2f} ({ci_l:.2f}–{ci_h:.2f})",
                # 'HR': hr,
                # 'CI_low': ci_l,
                # 'CI_high': ci_h,
                'p': p,
                'p_interaction': p_inter_before
            })

            # ----------------------
            # 📊 Cox（PSM后）
            # ----------------------
            hr, ci_l, ci_h, p = compute_cox(after_psm)

            after_results.append({
                'cohort': center,
                'stage': sub_group,
                'risk_group': risk_group,
                'No Chemo/Chemo(n)': f'{len(after_psm[after_psm["Chemotherapy"]==0])}/{len(after_psm[after_psm["Chemotherapy"]==1])}',
                'HR(95%CI)': f"{hr:.2f} ({ci_l:.2f}–{ci_h:.2f})",
                # 'HR': hr,
                # 'CI_low': ci_l,
                # 'CI_high': ci_h,
                'p': p,
                'p_interaction': p_inter_after
            })


# ==============================
# 💾 保存结果
# ==============================

before_df = pd.DataFrame(before_results)
after_df = pd.DataFrame(after_results)

save_path = os.path.join(SAVE_DIR, 'chemo_benefit_summary.xlsx')

with pd.ExcelWriter(save_path) as writer:
    before_df.to_excel(writer, sheet_name='before_psm', index=False)
    after_df.to_excel(writer, sheet_name='after_psm', index=False)

print(f'\n✅ 结果已保存: {save_path}')