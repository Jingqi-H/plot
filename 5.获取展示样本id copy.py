import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
SAVE_DIR = 'PLOTS/5.热图'
LABELS = ('Low Risk', 'High Risk')
# RESULT_DIR = 'patient_level_RandomTrainData'
# RESULT_DIR = '0.0001loadloss'
RESULT_DIR = 'final'



# =====================================================
# Data Load
# =====================================================
info_df = pd.read_csv(
    'ori_files/SXCH/clinical_info_all.csv',
    dtype={'case_id': str}
)

cutoff_df = pd.read_csv(f"PLOTS/2.km/{RESULT_DIR}/cutoff_media_risk_score.csv")
paths = glob.glob(f'results/moe_wsi/{RESULT_DIR}/summary_SXCH-Train_0.*.csv') + glob.glob(f'results/moe_wsi/{RESULT_DIR}/summary_SXCH-Val_0.*.csv')
print(len(paths), paths)

all_info = pd.read_csv('data_csv_2fold/SXCH/clinical_info_dummy_clean_multi_task.csv', dtype={'case_id': str})

for path in paths:
    if not 'Val' in path:
        continue
    center_name = find_center_name(path)
    survival_state = determine_survival_type(path)

    df = pd.read_csv(path)
    df['status'] = 1 - df['censorship']
    df['event_times'] = df['survival_time']

    # ================= ALL KM =================
    cutoff_all = cutoff_df.loc[cutoff_df['group'] == 'ALL', 'cutoff'].values[0]

    df['risk_group'] = (df['risk'] > cutoff_all).astype(int)

    final_df = pd.merge(df[['slide_id_wax', 'risk_group', 'risk']], all_info, on='slide_id_wax', how='left')
    final_df.to_csv(f'{SAVE_DIR}/{RESULT_DIR}/{center_name}-用于筛选样本.csv', index=False)
    print(f'结果保存到 {SAVE_DIR}/{RESULT_DIR}/{center_name}-用于筛选样本.csv')

