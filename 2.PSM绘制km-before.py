import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
import os
import matplotlib.image as mpimg

plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 15


# 532+680+789+927=2938

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
    # if title is not None:
    #     ax.set_title(title, fontsize=fontsiz)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
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


def merge_km_pngs(SAVE_DIR,
                  centers = ['SXCH-Train', 'SXCH-Val', 'External'],
                  sub_groups = ['All', 'Stage_2', 'Stage_3'],
                  risk_groups = [0,1],
                  output_format='png'):
    
    # centers = ['SXCH-Train', 'SXCH-Val', 'External']
    # sub_groups = ['All', 'Stage_2', 'Stage_3']
    # risk_groups = ['Low', 'High']  # ⚠️ 根据你实际情况修改

    for sub_group in sub_groups:

        fig, axes = plt.subplots(
            nrows=len(risk_groups),
            ncols=len(centers),
            figsize=(15, 8)
        )

        for i, risk in enumerate(risk_groups):
            for j, center in enumerate(centers):

                ax = axes[i, j]

                img_path = os.path.join(
                    SAVE_DIR,
                    f'Risk{risk}_{center}_{sub_group}.png'
                )

                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                else:
                    print(f'{img_path} 文件不存在')
                    ax.text(0.5, 0.5, 'Missing',
                            ha='center', va='center', fontsize=12)

                # ✅ 第一行写 center title
                if i == 0:
                    ax.set_title(center, fontsize=14)

                # ✅ 左侧写 risk
                if j == 0:
                    ax.set_ylabel(risk, fontsize=14)

                ax.axis('off')

        plt.tight_layout()

        save_path = os.path.join(
            SAVE_DIR,
            f'KM_Combined_{sub_group}.png'
        )

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'✅ Saved: {save_path}')


CUTOFF_TIME = 110
FONTSIZE = 15

# RESULT_DIR = '0.0001loadloss'
RESULT_DIR = 'final'

ROOT_DIR = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息'
SAVE_DIR = f'PLOTS/2.km/{RESULT_DIR}/化疗受益PSM-before'
os.makedirs(SAVE_DIR, exist_ok=True)

cutoff_df = pd.read_csv(f"PLOTS/2.km/{RESULT_DIR}/cutoff_media_risk_score.csv")
survival_state = 'OS'

for center in ['SXCH-Train+SXCH-Val', 'SXCH-Train', 'SXCH-Val', 'External']:
    
    if center in ['CMU1H', 'YYH', 'SYSUCC', 'TCGA_STAD']:
        info_df = pd.read_csv(f'{ROOT_DIR}/External_Info.csv', dtype={'case_id': str})
        info_df = info_df[info_df['center'] == center]
    elif center == 'SXCH-Train+SXCH-Val':
        info_df1 = pd.read_csv(f'{ROOT_DIR}/SXCH-Train_Info.csv', dtype={'case_id': str})
        info_df2 = pd.read_csv(f'{ROOT_DIR}/SXCH-Val_Info.csv', dtype={'case_id': str})
        info_df = pd.concat([info_df1, info_df2], axis=0)
    else:
        info_df = pd.read_csv(f'{ROOT_DIR}/{center}_Info.csv', dtype={'case_id': str})

    info_df.dropna(subset=['Chemotherapy'],inplace=True)
    info_df = info_df[(info_df['Stage'] != 1)]
    info_df = info_df[(info_df['Stage'] != 4)]
    
    info_df['censorship'] = 1 - info_df[survival_state+'_status']
    info_df['status'] = info_df[survival_state+'_status']
    info_df['event_times'] = info_df[survival_state]
    info_df['survival_time'] = info_df[survival_state]
    
    
    for sub_group in ['All', 'Stage_2', 'Stage_3']:
        print(f'处理{center} ({sub_group})')
        # if not (center == 'All_external' and sub_group == 'Stage_2'):
        #     continue

        df_cp = info_df.copy()
        if sub_group == 'All':
            cutoff_ = cutoff_df.loc[cutoff_df['group'] == 'ALL', 'cutoff'].values[0]
        else:
            subgroup_col, subgroup_value = sub_group.split('_', 1)
            # subgroup_cases = df_cp.loc[
            # df_cp[subgroup_col].astype(str) == subgroup_value,
            # 'case_id'
            # ]
            df_cp[subgroup_col] = df_cp[subgroup_col].apply(lambda x: round(x) if pd.notna(x) else x)
            df_cp[subgroup_col] = df_cp[subgroup_col].astype('Int64')
            subgroup_cases = df_cp.loc[df_cp[subgroup_col] == int(subgroup_value),'case_id']

            df_cp = df_cp[df_cp['case_id'].isin(subgroup_cases)].copy()
            cutoff_ = cutoff_df.loc[cutoff_df['group'] == sub_group, 'cutoff'].values[0]

        df_cp['risk_group'] = (df_cp['risk'] > cutoff_).astype(int)

        for risk_group in df_cp['risk_group'].unique():
            df_risk = df_cp[df_cp['risk_group'] == risk_group].copy()
            km_df = df_risk.copy()

            title_name = center if center != 'All_external' else 'External'
            plot_internal_km(
                    data_df=km_df,
                    labels=['No Chemo', 'Chemo'],
                    plot_type='Chemotherapy',
                    save_path=f'{SAVE_DIR}/Risk{risk_group}_{center}_{sub_group}.svg',
                    fontsiz=FONTSIZE,
                    cutoff_time=CUTOFF_TIME,
                    survival_state=survival_state,
                    title=title_name
                )
            
        # fdsfd

merge_km_pngs(SAVE_DIR,
            centers = ['SXCH-Train+SXCH-Val', 'SXCH-Train', 'SXCH-Val', 'External'],
            sub_groups = ['All', 'Stage_2', 'Stage_3'],
            risk_groups = ['0', '1'],
            output_format='png')
