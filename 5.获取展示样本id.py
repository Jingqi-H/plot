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
LABELS = ('Low Risk', 'High Risk')
# RESULT_DIR = 'patient_level_RandomTrainData'
# RESULT_DIR = '0.0001loadloss'
RESULT_DIR = 'final'
SAVE_DIR = 'PLOTS/5.热图/' + RESULT_DIR
os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================
# Data Load
# =====================================================
info_df = pd.read_csv(
    'ori_files/SXCH/clinical_info_all.csv',
    dtype={'case_id': str}
)

cutoff_df = pd.read_csv(f"PLOTS/2.km/{RESULT_DIR}/cutoff_media_risk_score.csv")


# center_name = 'SXCH-Train+SXCH-Val'
center_name = 'SXCH-Val'
path = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息/{center_name}_Info.csv'
survival_state = 'OS'

df = pd.read_csv(path)
df['status'] = df['OS_status']
df['censorship'] = 1 - df['OS_status']
df['survival_time'] = df['OS']
df['event_times'] = df['survival_time']

# ================= ALL KM =================
cutoff_all = cutoff_df.loc[cutoff_df['group'] == 'ALL', 'cutoff'].values[0]
df['risk_group'] = (df['risk'] > cutoff_all).astype(int)


# 保存到excel的sheet中
excel_writer = pd.ExcelWriter(f'{SAVE_DIR}/{center_name}-用于筛选样本.xlsx')

df.to_excel(excel_writer, sheet_name='ALL', index=False)
# 筛选出risk_group=0, Histo Type=1，Lauren Type=1,Stage=1， Grade=1的样本id
df1 = df[(df['risk_group'] == 0) & (df['Histo Type'] == 1) & (df['Lauren Type'] == 1) & (df['Stage'] == 1) & (df['Grade'] == 1)]
df1.to_excel(excel_writer, sheet_name='L', index=False)

# 筛选出risk_group=1, Histo Type=2，Lauren Type=2,Stage=3/4， Grade=2的样本id
df2 = df[(df['risk_group'] == 1) & (df['Histo Type'] == 2) & (df['Lauren Type'] == 2) & (df['Stage'].isin([3, 4])) & (df['Grade'] == 2)]
df2.to_excel(excel_writer, sheet_name='H', index=False)

excel_writer.close()

print(f'结果保存到 {SAVE_DIR}/{center_name}-用于筛选样本.xlsx')

