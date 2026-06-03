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
    
    print('缺失的列:', columns_to_impute)
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
def run_cox_and_score(train_df, cox_df, covariate_dict, save_path):
    """
    对 selected_covariates 中每个组合：
    - 在训练集训练 Cox 模型
    - 在验证集计算 risk score
    - 将训练集 summary 和验证集 risk score 保存到 Excel 不同 sheet
    """
    # writer = pd.ExcelWriter(save_path, engine='openpyxl')
    writer_val = pd.ExcelWriter(save_path, engine='openpyxl')
    writer_contri = pd.ExcelWriter(save_path.replace('cox_results_', '各个临床因素贡献度'), engine='openpyxl')

    for name, covariates in covariate_dict.items():
        # 训练 Cox 模型
        cph = CoxPHFitter(penalizer=0.1)
        temp_df = train_df[['survival_time', 'status'] + covariates].copy().dropna()
        cph.fit(temp_df, duration_col='survival_time', event_col='status')
        
        # 保存训练集 Cox summary
        cox_df_copy = cox_df.copy().dropna(subset= covariates)
        cox_df_copy['risk_score'] = cph.predict_partial_hazard(cox_df_copy[covariates])
        # if name == 'GRASP':
        #     cox_df_copy['risk_score'] = cox_df_copy['risk'].copy()
        # 只保存必要列
        risk_score_df = cox_df_copy[['case_id', 'risk_score', 'survival_time', 'status']]
        risk_score_df.to_excel(writer_val, sheet_name=f'{name}', index=False)

        '贡献'
        if name in ['Clinical','Clinical+GRASP']:
            # 从Cox模型summary中提取每个变量的Wald卡方值
            z_values = cph.summary['z'].values          # 提取z统计量
            wald_chi2 = np.square(z_values)             # χ² = z²（等价于Wald卡方值）
            total_chi2 = np.sum(wald_chi2)              # 所有变量总卡方值
            
            # 3. 计算论文要求的「χ² proportion test」相对贡献度
            relative_contribution = (wald_chi2 / total_chi2) * 100  # 百分比
            
            # 4. 整理贡献度结果（标注z值→χ²的转换，便于论文说明）
            contribution_df = pd.DataFrame({
                'Covariate': covariates,
                'Z_statistic': z_values,                # 原始z值
                'Wald_χ²(z²)': wald_chi2,               # 转换后的Wald卡方值
                'Total_χ²': total_chi2,                 # 总卡方值
                'Relative_Contribution(%)': relative_contribution,  # 论文的相对贡献度
                'P_value': cph.summary['p'].values      # P值（统计学显著性）
            })
            contribution_df.to_excel(writer_contri, sheet_name=f'{name}', index=False)

    writer_val.save()
    writer_contri.save()
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


def read_file(_file, _columns, rename_columns, cutoff_all, center=None):
    df = pd.read_csv(_file, dtype={'case_id': str})
    # if center == 'External':
    #     df = df[df['center'] != 'TCGA_STAD']
    
    df = df[_columns]
    df = df.rename(columns=rename_columns)
    df = prepare_multivariate_df(df, cutoff_all)
    # _cox_df = df[df['center'] == center]
    _cox_df = df.drop(columns=['center'])
    return _cox_df


# ============================================================
# 3️⃣ 主流程
# ============================================================
# exp_id = '0.0001loadloss'
exp_id = 'final'
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

files = ['SXCH-Train', 'SXCH-Val', 'External', 'YYH', 'SYSUCC', 'CMU1H']
root_dir = f'PLOTS/2.km/{exp_id}/临床&Risk信息'
for center in files:
    train_columns = _columns.copy()
    val_columns = _columns.copy()
    
    print(f'\n====================当前处理中心：{center}')
    _file = f'{root_dir}/{center}_Info.csv'
    _cox_df = read_file(_file, val_columns, rename_columns, cutoff_all, center)
    train_df = read_file(f'{root_dir}/SXCH-Train_Info.csv', train_columns, rename_columns, cutoff_all)


    # ---------- 定义联合建模的变量组合 ----------
    selected_covariates = {
        'Age': ['Age'],
        'CEA': ['CEA'],
        'Location': ['Location'],
        'pT': ['pT'],
        'pN': ['pN'],
        'pM': ['pM'],
        'Stage': ['Stage'],
        'Clinical': ['Age', 'CEA','Location', 'pT', 'pN', 'pM'],
        'GRASP': ['risk'],
        'Clinical+GRASP': ['Age', 'CEA','Location', 'pT', 'pN', 'pM', 'risk']
    }


    # ---------- 执行联合建模并保存结果 ----------
    save_path = f'PLOTS/4.联合建模/cox_results_{center}.xlsx'
    run_cox_and_score(train_df, _cox_df, selected_covariates, save_path)

