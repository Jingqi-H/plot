import os
import sys
# 添加项目根目录到Python路径，为了import自己的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index



# ============================================================
#  构建 Cox 用多因素数据
# ============================================================
def prepare_multivariate_df(df, exclude_cols=['case_id', 'OS']):
    """
    1 连续变量二值化
    2 TNM 分组合并
    3 One-hot 编码分类变量
    """
    cox_df = df.copy()

    cox_df['Age'] = (cox_df['Age'] >= 65).astype(int)
    cox_df['Grade'] = cox_df['Grade'].map({1:1,2:1,3:2})

    num_cols = cox_df.select_dtypes(include=[np.number]).columns
    # 需要转换的列 = 数值列 - 排除列
    cols_to_convert = [col for col in num_cols if col not in exclude_cols]
    # 转换为 pandas 可空整数类型（不会因为 NaN 报错）
    cox_df[cols_to_convert] = cox_df[cols_to_convert].astype('Int64')


    cox_df = pd.get_dummies(
        cox_df,
        columns=['Age', 'pT', 'pN',  'Gender', 'Location',  'HistologicalType',
       'LaurenType', 'Chemotherapy', 'Stage', 'Grade'],
        drop_first=True
    )

    return cox_df


# ============================================================
#  合并 Cox 临床数据
# ============================================================
def merge_data(
    info_df,
    result_df,
    cutoff,
):
    """
    1 合并 slide → patient
    2 聚合 risk score
    3 构造 survival_time / status / risk_group
    """
    
    

    result_df['event_times'] = result_df['survival_time'].copy()
    result_df['status'] = 1 - result_df['censorship']
    result_df['risk_group'] = (result_df['risk'] > cutoff).astype(int)

    final_df = pd.merge(
        result_df[['case_id', 'slide_id', 'slide_id_wax', 'risk', 'censorship', 'survival_time', 'event_times', 'status', 'risk_group']],
        info_df,
        on='case_id',
        how='inner'
    )

    # final_df.drop(columns=['case_id'], inplace=True)
    final_df.dropna(subset=['survival_time', 'status'], inplace=True)

    return final_df


exp_id = 'patient_level_RandomTrainData'
RESULT_DIR = f"results/moe_wsi/{exp_id}"
INFO_PATH = "ori_files/SXCH/clinical_info_all.csv"
CUTOFF_PATH = f"PLOTS/2.km/{exp_id}/cutoff_media_risk_score.csv"


# ---------- 1 读取并预处理临床数据 ----------
raw_df = pd.read_csv(INFO_PATH)
raw_df.drop_duplicates('case_id', inplace=True)
raw_df = raw_df[['case_id', 'Age', 'Gender', 'Location', 'Grade', 'Histo Type',
       'Lauren Type', 'Chemotherapy', 'Stage', 'Pathological T stage', 'Pathological N stage', 'Metastasis',
        'OS', 'OS_status']] # 'Microsatellite status', 'HER-2 status', 'DFS', 'DFS_status',
raw_df.rename(columns={'Histo Type': 'HistologicalType', 'Lauren Type': 'LaurenType',
                         'Pathological T stage': 'pT', 'Pathological N stage': 'pN', 'Metastasis': 'pM'}, inplace=True)
cox_base_df = prepare_multivariate_df(raw_df)

# ---------- 2 合并 数据 ----------
#读取文件summary_SXCH-Train_0.*.csv
file_path = glob.glob(f'{RESULT_DIR}/summary_SXCH-Train_0.*.csv')
result_df = pd.read_csv(file_path[0],dtype={'case_id': str})

cutoff_df = pd.read_csv(CUTOFF_PATH)
cutoff_all = cutoff_df.loc[cutoff_df['group'] == 'ALL', 'cutoff'].values[0]
final_df = merge_data(
        cox_base_df,
        result_df,
        cutoff_all
    )


'处理数据，让每个变量数值化：插补、映射——这个中心没有缺失的'
missing_counts = final_df.isnull().sum()
columns_to_impute = missing_counts[missing_counts > 0].index.tolist()
print('先查看缺失的列，再手工调整连续和分类变量:',columns_to_impute)

# categorical_cols = ['Tstage', 'Nstage', 'Mstage', 'TNMstage']
# continuous_cols = []

# # df_impute_missing_values = impute_missing_values(df, continuous_cols, categorical_cols)
# df_impute_missing_values = df.copy()


rename_dict = {
    'risk': f'Risk',
    'risk_group': 'Risk (H v L)',
    'Age_1': f"Age\n(\u226565 v 65)",
    'Gender_1': f'Gender\n(Male v Female)',
    'Location_2': f'Location\n(Body v Cardia)', 
    'Location_3': f'Location\n(Antrum v Cardia)', 
    'Location_4': f'Location\n(Whole v Cardia)', 
    'pT_2': f'pT\n(pT2 v pT1)', 
    'pT_3': f'pT\n(pT3 v pT1)', 
    'pT_4': f'pT\n(pT4 v pT1)', 
    'pN_1': f'pN\n(pN1 v pN0)', 
    'pN_2': f'pN\n(pN2 v pN0)', 
    'pN_3': f'pN\n(pN3 v pN0)', 
    'pM': f'pM\n(M1 vM0)', 
    'LaurenType_2': f'Lauren Type\n(Diffuse v Intesinal)', 
    'LaurenType_3': f'Lauren Type\n(Mixed v Intesinal)', 
    'Grade_2': f'Grade\n(Poorly v Moderately/Well)',
    # 'Grade_2': f'Grade\n(Moderately v Well)',
    # 'Grade_3': f'Grade\n(Poorly v Well)',
    'HistologicalType_2': f'Histo. Type\n(Other v Adenocarcinoma)', 
    'Chemotherapy_1': f'Chemotherapy (Yes v No)', 
    'Stage_2': f'Stage (II v I)', 
    'Stage_3': f'Stage (III v I)', 
    'Stage_4': f'Stage (IV v I)', 
}

'绘制ALL中，'