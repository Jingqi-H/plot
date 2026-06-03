import pandas as pd
import numpy as np
import glob
from lifelines import CoxPHFitter
import os
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer 


# ============================================================
# 1️⃣ 数据预处理
# ============================================================
def prepare_multivariate_df(df, exclude_cols=['case_id', 'OS']):
    """
    仅对连续数值变量做类型转换，不做哑变量。
    """
    cox_df = df.copy()
    cox_df['Age'] = (cox_df['Age'] >= 65).astype(int)  # 连续变量二值化
    num_cols = cox_df.select_dtypes(include=[np.number]).columns
    cols_to_convert = [col for col in num_cols if col not in exclude_cols]
    cox_df[cols_to_convert] = cox_df[cols_to_convert].astype('Int64')
    return cox_df


def merge_data(info_df, result_df, cutoff, is_ext=False):
    """
    合并临床信息与预测结果，生成 Cox 输入数据。
    """
    result_df['event_times'] = result_df['survival_time'].copy()
    result_df['status'] = 1 - result_df['censorship']
    result_df['risk_group'] = (result_df['risk'] > cutoff).astype(int)

    if not is_ext:
        target_df = result_df[['case_id', 'slide_id', 'slide_id_wax', 'risk',
                   'censorship', 'survival_time', 'event_times', 'status', 'risk_group']]
    else:
        temp_df = result_df[['case_id', 'slide_id', 'risk',
                   'censorship', 'survival_time', 'event_times', 'status', 'risk_group']]
        # 对于risk，要将一个患者的取平均
        risk_df = (
            temp_df.groupby("case_id")
              .agg({
                  "risk": "mean",
                #   "survival_time": "first",
                #   "censorship": "first"
              })
              .reset_index()
        )
        temp_df = temp_df.drop(columns=['risk']).drop_duplicates(subset=['case_id'])
        target_df = pd.merge(temp_df, risk_df, on='case_id', how='inner')
        
    final_df = pd.merge(
        target_df,
        info_df,
        on='case_id',
        how='inner'
    )
    final_df.dropna(subset=['survival_time', 'status'], inplace=True)
    return final_df


# ============================================================
# 2️⃣ 联合建模 & 验证集评分
# ============================================================
def run_cox_and_score(train_df, val_df, covariate_dict, save_path):
    """
    对 selected_covariates 中每个组合：
    - 在训练集训练 Cox 模型
    - 在验证集计算 risk score
    - 将训练集 summary 和验证集 risk score 保存到 Excel 不同 sheet
    """
    writer = pd.ExcelWriter(save_path, engine='openpyxl')

    for name, covariates in covariate_dict.items():
        # 训练 Cox 模型
        cph = CoxPHFitter(penalizer=0.1)
        temp_df = train_df[['survival_time', 'status'] + covariates].copy().dropna()
        cph.fit(temp_df, duration_col='survival_time', event_col='status')

        # 验证集 risk score
        val_df_copy = val_df.copy().dropna(subset= covariates)
        val_df_copy['risk_score'] = cph.predict_partial_hazard(val_df_copy[covariates])
        # 只保存必要列
        risk_score_df = val_df_copy[['case_id', 'risk_score', 'survival_time', 'status']]
        risk_score_df.to_excel(writer, sheet_name=f'{name}', index=False)

    writer.save()
    print(f"联合建模结果已保存至 {save_path}")

# def impute_missing_values(df, cols_to_impute, random_state=42, max_iter=10):

#     df_copy = df.copy()

#     # 只取要填补的列
#     subset = df_copy[cols_to_impute].apply(pd.to_numeric, errors='coerce')

#     # 定义 imputer
#     # imputer = IterativeImputer(random_state=random_state, max_iter=max_iter)
#     imputer = IterativeImputer(max_iter=max_iter, random_state=random_state, initial_strategy='most_frequent',)

#     # 拟合并填补
#     imputed = imputer.fit_transform(subset)

#     # 保留整数分类（四舍五入再转 int）
#     df_copy[cols_to_impute] = pd.DataFrame(imputed, 
#                                            columns=cols_to_impute,
#                                            index=subset.index).round().astype(int)

#     return df_copy

def impute_missing_values(
    df,
    continuous_cols=None,
    categorical_cols=None,
    random_state=0,
    max_iter=10
):
    df_copy = df.copy(deep=True)

    # ---------- 1. 连续变量：Iterative Imputer ----------
    if continuous_cols is not None and len(continuous_cols) > 0:
        cont_subset = df_copy[continuous_cols].apply(
            pd.to_numeric, errors='coerce'
        )

        # 防止全部是 NaN
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

    # ---------- 2. 分类 / TNM：众数填补 ----------
    if categorical_cols is not None and len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        # df_copy[categorical_cols] = cat_imputer.fit_transform(
        #     df_copy[categorical_cols]
        # )
        imputed = cat_imputer.fit_transform(df_copy[categorical_cols])

        df_copy[categorical_cols] = pd.DataFrame(
            imputed,
            columns=categorical_cols,
            index=df_copy.index
        )

    return df_copy



exp_id = '0.0001loadloss'

RESULT_DIR = f"results/moe_wsi/{exp_id}"
INFO_PATH = "ori_files/SXCH/clinical_info_all.csv"
CUTOFF_PATH = f"PLOTS/2.km/{exp_id}/cutoff_media_risk_score.csv"

columns = ['case_id', 'Age', 'Location', 'Pathological T stage',
                 'Pathological N stage', 'Metastasis', 'OS', 'OS_status']

# ---------- 读取临床数据 ----------
raw_df = pd.read_csv(INFO_PATH)
raw_df.drop_duplicates('case_id', inplace=True)
raw_df = raw_df[columns]

raw_df.rename(columns={
    'Histo Type': 'HistologicalType',
    'Lauren Type': 'LaurenType',
    'Pathological T stage': 'pT',
    'Pathological N stage': 'pN',
    'Metastasis': 'pM',
    # 'OS': 'survival_time',
    # 'OS_status': 'status'
}, inplace=True)

cox_base_df = prepare_multivariate_df(raw_df)  # 不做哑变量

# ---------- 读取 cutoff ----------
cutoff_df = pd.read_csv(CUTOFF_PATH)
cutoff_all = cutoff_df.loc[cutoff_df['group'] == 'ALL', 'cutoff'].values[0]

# ---------- 合并训练集和验证集 ----------
train_file = glob.glob(f'{RESULT_DIR}/summary_SXCH-Train_0.*.csv')[0]
train_df = pd.read_csv(train_file, dtype={'case_id': str})
train_cox_df = merge_data(cox_base_df, train_df, cutoff_all)


# ---------- 定义联合建模的变量组合 ----------
selected_covariates = {
    'Clinical': ['Age', 'Location', 'pT', 'pN', 'pM'],
    'GRASP': ['risk'],
    'Clinical+GRASP': ['Age', 'Location', 'pT', 'pN', 'pM', 'risk']
}

# ---------- 执行联合建模并保存结果 ----------
external_path = glob.glob('ori_files/*/clinical_info_all.csv')
for path_ in external_path:
    center = path_.split('/')[-2]
    if 'SXCH' in path_:
        continue
    print(f'处理{center}')
    ext_raw_df = pd.read_csv(path_, dtype={'case_id': str, 'slide_id': str})
    ext_raw_df.drop_duplicates('case_id', inplace=True)
    ext_raw_df = ext_raw_df[columns]

    ext_raw_df.rename(columns={
        'Pathological T stage': 'pT',
        'Pathological N stage': 'pN',
        'Metastasis': 'pM',
    }, inplace=True)
    # 填充缺失值
    column_impute = ['Location', 'pT', 'pN', 'pM']
    ext_raw_df[column_impute] = ext_raw_df[column_impute].apply(lambda x: x.fillna(x.mode()[0]))
    # ext_raw_df = impute_missing_values(ext_raw_df, categorical_cols=column_impute)

    ext_cox_df = prepare_multivariate_df(ext_raw_df) 

    val_file = glob.glob(f'{RESULT_DIR}/results_external/summary_{center}_0.*.csv')[0]
    val_df = pd.read_csv(val_file, dtype={'case_id': str, 'slide_id': str})

    val_cox_df = merge_data(ext_cox_df, val_df, cutoff_all, is_ext=True)
    save_path = f'PLOTS/4.联合建模/ext/{center}_result.xlsx'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    run_cox_and_score(train_cox_df, val_cox_df, selected_covariates, save_path)
    # break


