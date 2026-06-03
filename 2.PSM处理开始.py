import glob
import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def compute_smd(df, col, treatment='Chemotherapy'):
    treated = df[df[treatment]==1][col]
    control = df[df[treatment]==0][col]

    return abs(treated.mean() - control.mean()) / np.sqrt(
        (treated.var() + control.var()) / 2
    )
def propensity_score_matching(df, caliper=0.2, age_caliper=5):
    # treated = df[df['Chemotherapy'] == 1].copy()
    # control = df[df['Chemotherapy'] == 0].copy()
    treated = df[df['Chemotherapy'] == df['Chemotherapy'].value_counts().idxmax()].copy()
    control = df[df['Chemotherapy'] == df['Chemotherapy'].value_counts().idxmin()].copy()

    treated = treated.sort_values('ps')
    control = control.sort_values('ps')

    matched_treated_idx = []
    matched_control_idx = []
    control_used = set()
    for i, t_row in treated.iterrows():
        t_ps = t_row['ps']
        t_age = t_row['Age']

        # 只用未匹配的control
        available_control = control.loc[~control.index.isin(control_used)]

        if available_control.shape[0] == 0:
            break

        ps_diff = abs(available_control['ps'] - t_ps)
        sorted_idx = ps_diff.sort_values().index

        matched = False

        for idx in sorted_idx:
            c_ps = control.loc[idx, 'ps']
            c_age = control.loc[idx, 'Age']

            if abs(c_ps - t_ps) <= caliper and abs(c_age - t_age) <= age_caliper:
                matched_treated_idx.append(i)
                matched_control_idx.append(idx)
                control_used.add(idx)
                matched = True
                break

        if not matched:
            continue

    n_matched = len(matched_treated_idx)
    print(f'✅ 匹配成功: {n_matched} 对')
    print(f'✅ 治疗组匹配率: {n_matched/len(treated):.2%}')
    print(f'✅ 对照组使用率: {n_matched/len(control):.2%}')

    # ===================== 核心需求：生成匹配CaseID表格 =====================
    # 提取治疗组和对照组的CaseID
    treated_ids = treated.loc[matched_treated_idx, 'case_id'].values # 为str
    control_ids = control.loc[matched_control_idx, 'case_id'].values # 为str

    # 创建匹配结果DataFrame（每一行=一对匹配样本）
    matched_result = pd.DataFrame({
        'Match_ID': range(1, n_matched + 1),  # 配对编号（1,2,3...）
        'Treated_CaseID': treated_ids,       # 化疗组病例ID
        'Control_CaseID': control_ids        # 非化疗组病例ID
    })

    # # ===================== 生成匹配后的完整数据集（用于后续生存分析） =====================
    # matched_treated_df = treated.loc[matched_treated_idx].copy()
    # matched_control_df = control.loc[matched_control_idx].copy()
    # matched_df = pd.concat([matched_treated_df, matched_control_df], ignore_index=True)

    # 返回两个关键结果：匹配ID表 + 匹配后总数据集
    return matched_result

# def propensity_score_matching(df, caliper=0.03):

#     treated = df[df['Chemotherapy'] == 1].copy()
#     control = df[df['Chemotherapy'] == 0].copy()

#     matched_treated_idx = []
#     matched_control_idx = []

#     for i, t_row in treated.iterrows():
#         t_ps = t_row['ps']

#         candidates = control.copy()
#         candidates['ps_diff'] = abs(candidates['ps'] - t_ps)
#         candidates = candidates.sort_values('ps_diff')

#         for idx, c_row in candidates.iterrows():
#             if abs(c_row['ps'] - t_ps) <= caliper:
#                 matched_treated_idx.append(i)
#                 matched_control_idx.append(idx)
#                 break  # 找到一个就停

#     matched_df = pd.concat([
#         df.loc[matched_treated_idx],
#         df.loc[matched_control_idx]
#     ])

#     return matched_df

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
    return df_copy

def new_group(df, sub_group, cutoff_df):
    df_cp = df.copy()
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
    return df_cp



def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



CUTOFF_TIME = 110
FONTSIZE = 15

# RESULT_DIR = '0.0001loadloss'
RESULT_DIR = 'final'
ROOT_DIR = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息'

seed_torch(seed=42)
cutoff_df = pd.read_csv(f"PLOTS/2.km/{RESULT_DIR}/cutoff_media_risk_score.csv")


_cols = ['Age', 'Gender', 'Location', 'Grade', 'Histo Type',
       'Lauren Type', 'Chemotherapy',
       'Stage', 'Pathological T stage', 'Pathological N stage', 'Metastasis']

files = ['SXCH-Train+SXCH-Val','SXCH-Train', 'SXCH-Val', 'External','CMU1H', 'YYH', 'SYSUCC'] # 

for file in files:
    
    output_path = f'{ROOT_DIR}/PSM_{file}.xlsx'
    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    for sub_group in ['All', 'Stage_2', 'Stage_3']: # 

        path = f'{ROOT_DIR}/{file}_Info.csv'
        info_df = pd.read_csv(path, dtype={'case_id': str})
        info_df = info_df[(info_df['Stage'] != 1)]
        info_df = info_df[(info_df['Stage'] != 4)]
        if file == 'External':
            info_df = info_df[info_df['center']!='TCGA_STAD']

        info_df = new_group(info_df, sub_group, cutoff_df)
        for risk_group in info_df['risk_group'].unique():
            print(f'==============处理 {file}, {sub_group}, risk group{risk_group}')
            df = info_df[info_df['risk_group'] == risk_group]
            df = df.dropna(subset=['Chemotherapy'])

            # ========================
            # subgroup筛选（修复bug🔥）
            # ========================
            current_cols = _cols.copy()
            if 'Stage' in current_cols:
                current_cols.remove('Stage')

            # if sub_group != 'All':
            #     sub_g, sub_v = sub_group.split('_')
            #     sub_v = int(sub_v)
            #     df = df[df['Stage'] == sub_v]

            # ========================
            # 缺失值处理
            # ========================
            df_psm = df.copy()
            df_psm['Chemotherapy'] = df_psm['Chemotherapy'].astype(int)

            # missing_counts = df_psm.isnull().sum()
            # columns_to_impute = missing_counts[missing_counts > 0].index.tolist()
            # remove_cols = ['DFS', 'DFS_status', 'OS', 'OS_status',
            #             'Microsatellite status', 'HER-2 status', 'PFS', 'PFS_status']
            # columns_to_impute = [col for col in columns_to_impute if col not in remove_cols]
            # print('缺失的列:', columns_to_impute)
            # df_psm = impute_missing_values(df_psm, columns_to_impute)

            # ========================
            # 构建PS模型
            # ========================
            covariates = current_cols.copy()
            covariates.remove('Chemotherapy')

            # 🔥 加 Age²（关键改进）
            df_psm['Age2'] = df_psm['Age'] ** 2

            X = df_psm[covariates + ['Age2']]
            y = df_psm['Chemotherapy']

            # one-hot（排除Age）
            one_hot_cols = [col for col in covariates if col != 'Age']
            X_encoded = pd.get_dummies(X, columns=one_hot_cols, drop_first=True)

            # 🔥 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)

            # ========================
            # 训练PS模型
            # ========================
            ps_model = LogisticRegression(max_iter=1000, penalty='l2')
            ps_model.fit(X_scaled, y)

            df_psm['ps'] = ps_model.predict_proba(X_scaled)[:, 1]

            # ========================
            # Matching
            # ========================
            save_df = propensity_score_matching(
                df_psm,
                caliper=0.02,
                age_caliper=5   # 🔥 可调：3/5/10
            )

            save_df.to_excel(writer, sheet_name=f'{sub_group}#{risk_group}', index=False)


    writer.close()
    print(f"已保存到: {output_path}")
        # break

# 合并外部'CMU1H', 'YYH', 'SYSUCC',每个sheet的结果
output_path = f'{ROOT_DIR}/PSM_External(merged).xlsx'
writer = pd.ExcelWriter(output_path, engine='openpyxl')

_path1 = f'{ROOT_DIR}/PSM_CMU1H.xlsx'
_path2 = f'{ROOT_DIR}/PSM_YYH.xlsx'
_path3 = f'{ROOT_DIR}/PSM_SYSUCC.xlsx'

for sub_group in ['All', 'Stage_2', 'Stage_3']:
    for risk_group in [0, 1]:
        df1 = pd.read_excel(_path1, sheet_name=f'{sub_group}#{risk_group}')
        df2 = pd.read_excel(_path2, sheet_name=f'{sub_group}#{risk_group}')
        df3 = pd.read_excel(_path3, sheet_name=f'{sub_group}#{risk_group}')
        df = pd.concat([df1, df2, df3], axis=0)
        # 获取 新的New_Match_ID
        df['New_Match_ID'] = range(1, len(df) + 1)
        df.to_excel(writer, sheet_name=f'{sub_group}#{risk_group}', index=False)

writer.close()
