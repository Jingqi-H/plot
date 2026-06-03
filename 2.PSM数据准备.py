import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
        result = result.replace('_dfs', '-DFS')
    if 'pfs' in result:
        result = result.replace('_pfs', '-PFS')
    
    if 'slide' in result:
        result = result.replace('_slide', '')

    return result


def determine_survival_type(path):
    if 'dfs' in path:
        return 'DFS'
    if 'pfs' in path:
        return 'PFS'
    return 'OS'

def feats_stardand(df, featsName=None): 
    df[featsName] = ( df[featsName] + 0.5 ).astype(int) 
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
        df_copy['Lauren Type'] = df_copy['Lauren Type'].map({0:1,1:1,2:2,3:3,4:3})
        df_copy['Location'] = df_copy['Location'].map({0:1,1:1,2:2,3:3,4:4,5:4})
        df_copy['Grade'] = df_copy['Grade'].map({0:1,1:1,2:2,3:2})
    return df_copy


CUTOFF_TIME = 110
FONTSIZE = 15
RESULT_DIR = 'final'
# RESULT_DIR = '0.0001loadloss'
SAVE_DIR = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息'
os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================
# Data Load
# =====================================================
paths = glob.glob(f'results/moe_wsi/{RESULT_DIR}/summary_SXCH-Train_*.csv') + \
        glob.glob(f'results/moe_wsi/{RESULT_DIR}/summary_SXCH-Val_*.csv') + \
        glob.glob(f'results/moe_wsi/{RESULT_DIR}/results_external/summary_*.csv')
print(f'Number of paths: {len(paths)}')

# =====================================================
# Main Loop
# =====================================================
for path in paths:
    center_name = find_center_name(path)
    print('=============================', center_name)
    if center_name != 'TCGA_STAD':
        continue

    info_df = pd.read_csv(f'ori_files/{center_name.split("-")[0]}/clinical_info_all.csv',dtype={'case_id': str})
    info_df.drop_duplicates(subset=['case_id'], inplace=True)

    df = pd.read_csv(path,dtype={'case_id': str})
    agg_df = (
        df.groupby("case_id").agg({ "risk": "mean",}).reset_index()
    )

    final_df = pd.merge(agg_df, info_df, on='case_id', how='left')
    final_df['center'] = center_name
    final_df['Grade'] = final_df['Grade'].map({1: 1, 2: 1, 3: 2})

    missing_counts = final_df.isnull().sum()
    columns_to_impute = missing_counts[missing_counts > 0].index.tolist()
    remove_cols = ['DFS', 'DFS_status', 'OS', 'OS_status',
                'Microsatellite status', 'HER-2 status', 'PFS', 'PFS_status']
    columns_to_impute = [col for col in columns_to_impute if col not in remove_cols]
    print('缺失的列:', columns_to_impute)
    final_df = impute_missing_values(final_df, columns_to_impute)
    # print(final_df['CEA'].unique())

    final_df.to_csv(f'{SAVE_DIR}/{center_name}_Info.csv', index=False)
    print(f'{center_name}有{len(final_df)}个样本')

# 合并外部队列
all_df = pd.DataFrame()
for center in ['CMU1H','YYH', 'SYSUCC', 'TCGA_STAD']: # , 'TCGA_STAD'
    df = pd.read_csv(f'{SAVE_DIR}/{center}_Info.csv',dtype={'case_id': str})
    all_df = pd.concat([all_df, df], axis=0)
all_df.to_csv(f'{SAVE_DIR}/External_Info.csv', index=False)

all_df = pd.DataFrame()
for center in ['CMU1H','YYH', 'SYSUCC', 'TCGA_STAD']: # , 'TCGA_STAD'
    df = pd.read_csv(f'{SAVE_DIR}/{center}-DFS_Info.csv',dtype={'case_id': str})
    all_df = pd.concat([all_df, df], axis=0)
all_df.to_csv(f'{SAVE_DIR}/External-DFS_Info.csv', index=False)



all_df = pd.DataFrame()
for center in ['SXCH-Train', 'SXCH-Val']:
    df = pd.read_csv(f'{SAVE_DIR}/{center}_Info.csv',dtype={'case_id': str})
    all_df = pd.concat([all_df, df], axis=0)
all_df.to_csv(f'{SAVE_DIR}/SXCH-Train+SXCH-Val_Info.csv', index=False)


all_df = pd.DataFrame()
for center in ['SXCH-Train-DFS', 'SXCH-Val-DFS']:
    df = pd.read_csv(f'{SAVE_DIR}/{center}_Info.csv',dtype={'case_id': str})
    all_df = pd.concat([all_df, df], axis=0)
all_df.to_csv(f'{SAVE_DIR}/SXCH-Train-DFS+SXSXCH-Val-DFS_Info.csv', index=False)