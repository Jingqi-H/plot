import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
# =====================================================
# Global Config
# =====================================================
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 15


def compute_smd(df, col, treatment='Chemotherapy'):
    treated = df[df[treatment]==1][col]
    control = df[df[treatment]==0][col]

    return abs(treated.mean() - control.mean()) / np.sqrt(
        (treated.var() + control.var()) / 2
    )

# def propensity_score_matching(df, caliper=0.05):

#     treated = df[df['Chemotherapy'] == 1].copy()
#     control = df[df['Chemotherapy'] == 0].copy()

#     treated = treated.sort_values('ps')
#     control = control.sort_values('ps')

#     matched_treated_idx = []
#     matched_control_idx = []

#     control_used = set()

#     for i, t_row in treated.iterrows():
#         t_ps = t_row['ps']

#         # 计算所有control的差距
#         control['ps_diff'] = abs(control['ps'] - t_ps)

#         # 找最近邻
#         candidate = control.loc[control['ps_diff'].idxmin()]

#         # caliper限制
#         if abs(candidate['ps'] - t_ps) <= caliper and candidate.name not in control_used:
#             matched_treated_idx.append(i)
#             matched_control_idx.append(candidate.name)
#             control_used.add(candidate.name)

#     matched_df = pd.concat([
#         df.loc[matched_treated_idx],
#         df.loc[matched_control_idx]
#     ])

#     return matched_df

def propensity_score_matching(df, caliper=0.03):

    treated = df[df['Chemotherapy'] == 1].copy()
    control = df[df['Chemotherapy'] == 0].copy()

    matched_treated_idx = []
    matched_control_idx = []

    for i, t_row in treated.iterrows():
        t_ps = t_row['ps']

        candidates = control.copy()
        candidates['ps_diff'] = abs(candidates['ps'] - t_ps)
        candidates = candidates.sort_values('ps_diff')

        for idx, c_row in candidates.iterrows():
            if abs(c_row['ps'] - t_ps) <= caliper:
                matched_treated_idx.append(i)
                matched_control_idx.append(idx)
                break  # 找到一个就停

    matched_df = pd.concat([
        df.loc[matched_treated_idx],
        df.loc[matched_control_idx]
    ])

    return matched_df


CUTOFF_TIME = 110
FONTSIZE = 15

RESULT_DIR = '0.0001loadloss'
ROOT_DIR = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息'

_cols = ['Age', 'Gender', 'Location', 'Grade', 'Histo Type',
       'Lauren Type', 'Chemotherapy',
       'Stage', 'Pathological T stage', 'Pathological N stage', 'Metastasis']


for sub_group in ['All', 'Stage_2', 'Stage_3']:
    # if sub_group == 'All':
    #     continue
    print(f'======================当前处理:{sub_group}')
    output_path = f'{ROOT_DIR}/PSM_{sub_group}.xlsx'
    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    
    

    files = ['SXCH-Train', 'SXCH-Val', 'All_external'] # 不处理TCGA_STAD
    for file in files:
        if file == 'TCGA_STAD' and sub_group != 'All':
            continue
        print('\n当前处理:', file)
        path = f'{ROOT_DIR}/{file}_Info.csv'
        
        info_df = pd.read_csv(path, dtype={'case_id': str})

        df = info_df.copy()
        df = df.dropna(subset=['Chemotherapy'])  # treatment必须存在
        if sub_group != 'All':
            sub_g, sub_v = sub_group.split('_')
            sub_v = int(sub_v)
            df = df[df['Stage'] == sub_v]
            _cols = [col for col in _cols if col != 'Stage']
        
        target_cols = _cols.copy()
        if 'TCGA_STAD' in file:
            target_cols.remove('Histo Type')
            target_cols.remove('Lauren Type')
        
        # 复制数据
        df_psm = df.copy()
        df_psm['Chemotherapy'] = df_psm['Chemotherapy'].astype(int)

        # 需要做 --------- 缺失值填补 ----------？？
        missing_counts = df_psm.isnull().sum()
        columns_to_impute = missing_counts[missing_counts > 0].index.tolist()
        remove_cols = ['DFS', 'DFS_status', 'OS', 'OS_status','Microsatellite status', 'HER-2 status']
        if 'TCGA_STAD' in file:
            remove_cols += ['Histo Type', 'Lauren Type']
        columns_to_impute = [col for col in columns_to_impute if col not in remove_cols]
        print('缺失的列:',columns_to_impute)
        df_psm[columns_to_impute] = df_psm[columns_to_impute].apply(lambda x: x.fillna(x.mode()[0]))
        # cat_imputer = SimpleImputer(strategy='most_frequent')
        # df_psm[columns_to_impute] = cat_imputer.fit_transform(df_psm[columns_to_impute])

        covariates = target_cols.copy()
        covariates.remove('Chemotherapy')

        X = df_psm[covariates]
        y = df_psm['Chemotherapy']

        one_hot_cols = covariates.copy()
        one_hot_cols.remove('Age')
        X_encoded = pd.get_dummies(X, columns=one_hot_cols, drop_first=True) 

        ps_model = LogisticRegression(max_iter=1000, penalty='l2')
        ps_model.fit(X_encoded, y)

        # propensity score
        '计算ps得分'
        df_psm['ps'] = ps_model.predict_proba(X_encoded)[:, 1]
        '根据ps得分,采用(Nearest Neighbor+Caliper Matching)匹配,回归调整,IPTW（Inverse Probability Weighting）加权,分层等形式均衡两组协变量差异'
        matched_df = propensity_score_matching(df_psm, caliper=0.03)

        print("原始样本数:", len(df_psm))
        print("匹配后样本数:", len(matched_df))

        # ================================
        # 保存 matched case_id + ps
        # ================================
        save_df = matched_df[['case_id', 'ps']].copy()
        save_df.to_excel(writer, sheet_name=file, index=False)
    writer.close()
    print(f"已保存到: {output_path}")
    # break