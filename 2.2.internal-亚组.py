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
    save_path=None,   # ⭐ 可选
    ax=None,          # ⭐ 新增
    fontsiz=14,
    survival_state='OS',
    cutoff_time=None,
):
    if ax is None:
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

        ax.step(sf.index, sf[labels[g]], where='post',
                linewidth=2.5, color=colors[g])

        ax.fill_between(
            ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
            color=colors[g], alpha=0.15, step='post'
        )

        kmfs.append(kmf)

    # ----- Style -----
    ax.set_ylim(0.05, 1.03)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel('Time (months)', fontsize=fontsiz)
    ax.set_ylabel(survival_state, fontsize=fontsiz)
    ax.tick_params(axis='both', labelsize=fontsiz)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ----- Statistics -----
    group0 = data_df[data_df[plot_type] == 0]
    group1 = data_df[data_df[plot_type] == 1]

    stat_text = new_p_value(group1, group0, data_df, group_type=plot_type)

    ax.text(0.05, 0.05, stat_text,
            transform=ax.transAxes,
            fontsize=fontsiz-2)

    # ⭐ 只有单图才加 risk table
    if save_path is not None:
        add_at_risk_counts(*kmfs, ax=ax, rows_to_show=['At risk'])

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

def get_subgroup_df(df, group_name, cutoff):
    subgroup_col, subgroup_value = group_name.split('_', 1)
    subgroup_value = int(float(subgroup_value))

    df_sub = df[df[subgroup_col] == subgroup_value].copy()

    if len(df_sub) < 10:
        return None

    df_sub['risk_group'] = (df_sub['risk'] > cutoff).astype(int)
    return df_sub

def generate_ALL_km(df, cutoff_df, save_dir, center_name):
    subgroup_cutoffs = cutoff_df[cutoff_df['group'] == 'ALL']

    for _, row in subgroup_cutoffs.iterrows():
        group_name = row['group']
        cutoff = row['cutoff']

        df_sub = df.copy()

        if len(df_sub) < 10:
            return None

        df_sub['risk_group'] = (df_sub['risk'] > cutoff).astype(int)

        if df_sub is None:
            print(f'{group_name} 样本太少，跳过')
            continue

        # 单图保存
        group_name = group_map.get(group_name, group_name)
        plot_internal_km(
            data_df=df_sub,
            labels=list(LABELS),
            plot_type='risk_group',
            save_path=f'{save_dir}/{group_name}_{center_name}.svg',
            fontsiz=FONTSIZE,
            cutoff_time=CUTOFF_TIME,
            survival_state=survival_state
        )


def generate_subgroup_km(df, cutoff_df, save_dir, center_name):
    subgroup_cutoffs = cutoff_df[cutoff_df['group'] != 'ALL']

    subgroup_results = []

    for _, row in subgroup_cutoffs.iterrows():
        group_name = row['group']
        cutoff = row['cutoff']

        df_sub = get_subgroup_df(df, group_name, cutoff)

        if df_sub is None:
            print(f'{group_name} 样本太少，跳过')
            continue

        # 单图保存
        group_name = group_map.get(group_name, group_name)
        plot_internal_km(
            data_df=df_sub,
            labels=list(LABELS),
            plot_type='risk_group',
            save_path=f'{save_dir}/{group_name}_{center_name}.svg',
            fontsiz=FONTSIZE,
            cutoff_time=CUTOFF_TIME,
            survival_state=survival_state
        )

        subgroup_results.append((group_name, df_sub))

    return subgroup_results

import math

def plot_multi_subgroup_km(
    subgroup_results,
    save_path,
    n_cols=4,
    center_name=None,
):
    n = len(subgroup_results)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5*n_cols, 4*n_rows))

    axes = axes.flatten()

    for i, (group_name, df_sub) in enumerate(subgroup_results):
        ax = axes[i]

        plot_internal_km(
            data_df=df_sub,
            labels=list(LABELS),
            plot_type='risk_group',
            ax=ax,   # ⭐ 关键
            fontsiz=12,
            cutoff_time=CUTOFF_TIME,
            survival_state=survival_state
        )

        # ⭐ 标题更专业
        group_name = group_map.get(group_name, group_name)
        ax.set_title(f"{group_name}\n(n={len(df_sub)})", fontsize=11)

    # 删除空白图
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    if not center_name is None:
        plt.suptitle(
            f"{center_name}", 
            fontsize=16, 
            fontweight='bold',  # 加粗
            y=0.99,             # 垂直位置（0-1，越大越靠上）
            # pad=20              # 与子图的间距
        )


    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


CUTOFF_TIME = 110
FONTSIZE = 15
LABELS = ('Low Risk', 'High Risk')
# RESULT_DIR = 'patient_level_RandomTrainData'
# RESULT_DIR = '0.0001loadloss'
RESULT_DIR = 'final'
survival_state = 'OS'
SAVE_DIR = f'PLOTS/2.km/{RESULT_DIR}/{survival_state}'
os.makedirs(SAVE_DIR, exist_ok=True)

cutoff_path = f"PLOTS/2.km/{RESULT_DIR}/cutoff_media_risk_score.csv"
cutoff_df = pd.read_csv(cutoff_path)
df_template = pd.read_excel('PLOTS/@source/template_cutoff.xlsx')
group_map = dict(zip(df_template["group"], df_template["new"]))

# =====================================================
# Main Loop
# =====================================================
center_name = 'SXCH'
if survival_state != 'OS':
    path_train = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息/{center_name}-Train-{survival_state}_Info.csv'
    path_val = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息/{center_name}-Val-{survival_state}_Info.csv'
else:
    path_train = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息/{center_name}-Train_Info.csv'
    path_val = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息/{center_name}-Val_Info.csv'


df1 = pd.read_csv(path_train)
# df1['center'] = 'SXCH Training'
df2 = pd.read_csv(path_val)
# df2['center'] = 'SXCH Validation'
df = pd.concat([df1, df2], axis=0)

df['censorship'] = 1 - df[f'{survival_state}_status']
df['status'] = df[f'{survival_state}_status']
df['event_times'] = df[f'{survival_state}']
df['survival_time'] = df[f'{survival_state}']
df['Age'] = (df['Age'] > 65).astype(int) # 大于65为1, 否则为0

'Train+Val'
# Step 1: 单图 + 收集结果
center_name = 'SXCH' if survival_state != 'DFS' else 'SXCH-DFS'
subgroup_results = generate_subgroup_km(
    df=df,
    cutoff_df=cutoff_df,
    save_dir=SAVE_DIR,
    center_name=center_name
)
# Step 2: 总图（论文用）
plot_multi_subgroup_km(
    subgroup_results,
    save_path=f"{SAVE_DIR}/ALL_subgroups_{center_name}.svg",
    n_cols=4,
    center_name=center_name

)

'Train'
center_name = 'SXCH-Train' if survival_state != 'DFS' else 'SXCH-Train-DFS'
train_df = df[df['center'] == center_name]
# Step 1: 单图 + 收集结果
subgroup_results = generate_subgroup_km(
    df=train_df ,
    cutoff_df=cutoff_df,
    save_dir=SAVE_DIR,
    center_name=center_name
)
# Step 2: 总图（论文用🔥）
plot_multi_subgroup_km(
    subgroup_results,
    save_path=f"{SAVE_DIR}/ALL_subgroups_{center_name}.svg",
    n_cols=4,
    center_name=center_name
)
generate_ALL_km(train_df, cutoff_df, SAVE_DIR, center_name)

'Val'
center_name = 'SXCH-Val' if survival_state != 'DFS' else 'SXCH-Val-DFS'
val_df = df[df['center'] == center_name]
# Step 1: 单图 + 收集结果
subgroup_results = generate_subgroup_km(
    df=val_df,
    cutoff_df=cutoff_df,
    save_dir=SAVE_DIR,
    center_name=center_name
)
# Step 2: 总图（论文用🔥）
plot_multi_subgroup_km(
    subgroup_results,
    save_path=f"{SAVE_DIR}/ALL_subgroups_{center_name}.svg",
    n_cols=4,
    center_name=center_name
)
generate_ALL_km(val_df, cutoff_df, SAVE_DIR, center_name)
