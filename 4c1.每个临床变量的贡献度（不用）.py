import pandas as pd
import numpy as np
import glob
from lifelines import CoxPHFitter

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
        contribution_df.to_excel(writer, sheet_name=f'{name}', index=False)
        

    writer.save()
    print(f"联合建模结果已保存至 {save_path}")


# ============================================================
# 3️⃣ 主流程
# ============================================================
# exp_id = 'patient_level_RandomTrainData'
exp_id = '0.0001loadloss'

RESULT_DIR = f"results/moe_wsi/{exp_id}"
INFO_PATH = "ori_files/SXCH/clinical_info_all.csv"
CUTOFF_PATH = f"PLOTS/2.km/{exp_id}/cutoff_media_risk_score.csv"

# ---------- 读取临床数据 ----------
raw_df = pd.read_csv(INFO_PATH)
raw_df.drop_duplicates('case_id', inplace=True)
raw_df = raw_df[['case_id', 'Age', 'Gender', 'Location', 'Grade', 'Histo Type',
                 'Lauren Type', 'Chemotherapy', 'Stage', 'Pathological T stage',
                 'Pathological N stage', 'Metastasis', 'OS', 'OS_status']]

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
val_file = glob.glob(f'{RESULT_DIR}/summary_SXCH-Val_0.*.csv')[0]

train_df = pd.read_csv(train_file, dtype={'case_id': str})
val_df = pd.read_csv(val_file, dtype={'case_id': str})

train_cox_df = merge_data(cox_base_df, train_df, cutoff_all)
val_cox_df = merge_data(cox_base_df, val_df, cutoff_all)

# ---------- 定义联合建模的变量组合 ----------
selected_covariates = {
    'Clinical': ['Age', 'Location', 'pT', 'pN', 'pM'],
    'Clinical+GRASP': ['Age', 'Location', 'pT', 'pN', 'pM', 'risk']
}

# ---------- 执行联合建模并保存结果 ----------
save_path = 'PLOTS/4.联合建模/各个临床因素的贡献度.xlsx'
run_cox_and_score(train_cox_df, val_cox_df, selected_covariates, save_path)


