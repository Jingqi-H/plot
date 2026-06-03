import pandas as pd
import numpy as np
import glob
from lifelines import CoxPHFitter
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1️⃣ 数据预处理
# ============================================================
def prepare_multivariate_df(df, cutoff, exclude_cols=['case_id', 'OS', 'risk', 'center', 'survival_time', 'event_times']):
    """
    仅对连续数值变量做类型转换，不做哑变量。
    """
    df.dropna(subset=['OS', 'OS_status', 'risk'], inplace=True)
    df['survival_time'] = df['OS'].copy()
    df['censorship'] = 1 - df['OS_status'].copy()
    df['event_times'] = df['OS'].copy()
    df['status'] = df['OS_status'].copy()
    df['risk_group'] = (df['risk'] > cutoff).astype(int)

    cox_df = df.copy()
    cox_df['Age'] = (cox_df['Age'] >= 65).astype(int)  # 连续变量二值化
    num_cols = cox_df.select_dtypes(include=[np.number]).columns
    cols_to_convert = [col for col in num_cols if col not in exclude_cols]
    cox_df[cols_to_convert] = cox_df[cols_to_convert].astype('Int64')


    missing_counts = df.isnull().sum()
    columns_to_impute = missing_counts[missing_counts > 0].index.tolist()
    remove_cols = ['DFS', 'DFS_status', 'OS', 'OS_status',
                'Microsatellite status', 'HER-2 status', 'PFS', 'PFS_status']
    columns_to_impute = [col for col in columns_to_impute if col not in remove_cols]
    
    print('============缺失的列:', columns_to_impute)
    df = impute_missing_values(df, columns_to_impute)
    for col in columns_to_impute:
        print(col,df[col].unique().tolist())
    df['LaurenType'] = df['LaurenType'].map({0:1,1:1,2:2,3:3})
    
    return cox_df


def merge_data(info_df, result_df, cutoff):
    """
    合并临床信息与预测结果，生成 Cox 输入数据。
    """
    result_df['event_times'] = result_df['survival_time'].copy()
    result_df['status'] = 1 - result_df['censorship']
    result_df['risk_group'] = (result_df['risk'] > cutoff).astype(int)

    final_df = pd.merge(
        result_df[['case_id', 'slide_id', 'slide_id_wax', 'risk',
                   'censorship', 'survival_time', 'event_times', 'status', 'risk_group']],
        info_df,
        on='case_id',
        how='inner'
    )
    final_df.dropna(subset=['survival_time', 'status'], inplace=True)
    return final_df


# ============================================================
# 2️⃣ 联合建模 & 验证集评分
# ============================================================
def run_cox_and_score(train_df, covariate_dict, save_path):
    """
    对 selected_covariates 中每个组合：
    - 在训练集训练 Cox 模型
    - 在验证集计算 risk score
    - 将训练集 summary 和验证集 risk score 保存到 Excel 不同 sheet
    """
    # writer = pd.ExcelWriter(save_path, engine='openpyxl')
    writer_train = pd.ExcelWriter(save_path, engine='openpyxl')

    for name, covariates in covariate_dict.items():
        # 训练 Cox 模型
        cph = CoxPHFitter(penalizer=0.1)
        temp_df = train_df[['survival_time', 'status'] + covariates].copy().dropna()
        cph.fit(temp_df, duration_col='survival_time', event_col='status')
        
        # 保存训练集 Cox summary
        train_df_copy = train_df.copy().dropna(subset= covariates)
        train_df_copy['risk_score'] = cph.predict_partial_hazard(train_df_copy[covariates])
        # 只保存必要列
        risk_score_df = train_df_copy[['case_id', 'risk_score', 'survival_time', 'status']]
        risk_score_df.to_excel(writer_train, sheet_name=f'{name}', index=False)

    writer_train.save()
    print(f"联合建模结果已保存至 {save_path}")

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
    
    return df_copy

# ============================================================
# 3️⃣ 主流程
# ============================================================
exp_id = '0.0001loadloss'
CUTOFF_PATH = f"PLOTS/2.km/{exp_id}/cutoff_media_risk_score.csv"

# ---------- 读取 cutoff ----------
cutoff_df = pd.read_csv(CUTOFF_PATH)
cutoff_all = cutoff_df.loc[cutoff_df['group'] == 'ALL', 'cutoff'].values[0]


# ---------- 读取临床数据 ----------
_columns = ['case_id', 'risk','Age', 'Gender', 'CEA', 'CA199','Location', 'Grade', 'Histo Type',
                 'Lauren Type', 'Chemotherapy', 'Stage', 'Pathological T stage',
                 'Pathological N stage', 'Metastasis', 'OS', 'OS_status', 'center']
rename_columns = {
    'Histo Type': 'HistologicalType',
    'Lauren Type': 'LaurenType',
    'Pathological T stage': 'pT',
    'Pathological N stage': 'pN',
    'Metastasis': 'pM',
    # 'OS': 'survival_time',
    # 'OS_status': 'status'
}

train_file = 'PLOTS/2.km/0.0001loadloss/临床&Risk信息/SXCH-Train_Info.csv'
train_df = pd.read_csv(train_file, dtype={'case_id': str})
val_file = 'PLOTS/2.km/0.0001loadloss/临床&Risk信息/SXCH-Val_Info.csv'
val_df = pd.read_csv(val_file, dtype={'case_id': str})
df = pd.concat([train_df, val_df])
df = df[_columns]
df = df.rename(columns=rename_columns)
df = prepare_multivariate_df(df, cutoff_all)


# ---------- 定义联合建模的变量组合 ----------
selected_covariates = {
    'Age': ['Age'],
    'CEA': ['CEA'],
    'Location': ['Location'],
    'pT': ['pT'],
    'pN': ['pN'],
    'pM': ['pM'],
    'Clinical': ['Age', 'CEA', 'Location', 'pT', 'pN', 'pM'],
    'GRASP': ['risk'],
    'Clinical+GRASP': ['Age', 'CEA', 'Location', 'pT', 'pN', 'pM', 'risk']
}

train_cox_df = df[df['center'] == 'SXCH-Train']
train_cox_df = train_cox_df.drop(columns=['center'])

val_cox_df = df[df['center'] == 'SXCH-Val']
val_cox_df = val_cox_df.drop(columns=['center'])

# ---------- 执行联合建模并保存结果 ----------
save_path = 'PLOTS/4.联合建模/cox_results_train.xlsx'
run_cox_and_score(train_cox_df, selected_covariates, save_path)

save_path = 'PLOTS/4.联合建模/cox_results_val.xlsx'
run_cox_and_score(val_cox_df, selected_covariates, save_path)
