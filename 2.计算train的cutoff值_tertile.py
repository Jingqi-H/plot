import glob
import pandas as pd
import os


# =====================================================
# 1️⃣ 计算某个临床变量的亚组中位风险
# =====================================================
def compute_subgroup_tertile_cutoffs(
    train_df: pd.DataFrame,
    info_df: pd.DataFrame,
    subgroup_col: str,
    model_name: str
) -> pd.DataFrame:
    """
    根据某个临床变量 (subgroup_col)，
    计算每个亚组的中位风险分数。

    Returns
    -------
    DataFrame with columns:
        group | model | cutoff
    """

    results = []

    # 只保留需要的列，并去除缺失
    valid_info = info_df[['case_id', subgroup_col]].dropna()

    # 排序后的亚组列表
    subgroups = sorted(valid_info[subgroup_col].unique())

    for subgroup in subgroups:

        # 当前亚组的 case_id
        subgroup_cases = valid_info.loc[
            valid_info[subgroup_col] == subgroup,
            'case_id'
        ]

        # 在 train_df 中筛选对应样本
        subgroup_train = train_df[
            train_df['case_id'].isin(subgroup_cases)
        ]

        if subgroup_train.empty:
            continue

        cutoff1 = train_df['risk'].quantile(0.333)
        cutoff2 = train_df['risk'].quantile(0.667)

        results.append({
            'group': f'{subgroup_col}_{subgroup}',
            'model': model_name,
            'cutoff1': cutoff1,
            'cutoff2': cutoff2
        })

    return pd.DataFrame(results)



# ---------- 基础配置 ----------
# RESULT_DIR = 'patient_level_RandomTrainData'
RESULT_DIR = '0.0001loadloss'


INFO_PATH = 'ori_files/SXCH/clinical_info_all.csv'
TRAIN_GLOB = f'results/*/{RESULT_DIR}/summary_SXCH-Train_*.csv'
OUTPUT_PATH = f'PLOTS/2.km/{RESULT_DIR}_tertile/cutoff_tertile_risk_score.csv'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

RENAME_MODEL = {
    # 'amil_wsi': 'AMIL',
    'moe_wsi': 'MoE',
}

SUBGROUP_COLUMNS = [
    'Grade',
    'Location',
    'Stage',
    'Histo Type',
    'Lauren Type',
    'Chemotherapy',
    'Pathological T stage', 'Pathological N stage', 'Metastasis',
]

# ---------- 读取临床信息 ----------
info_df = pd.read_csv(INFO_PATH, dtype={'case_id': str})

# ---------- 查找所有训练结果 ----------
train_paths = glob.glob(TRAIN_GLOB)

all_results = []

for train_path in train_paths:

    model_key = train_path.split('/')[1]
    model_name = RENAME_MODEL.get(model_key)

    # 跳过 AMIL
    if model_name == 'AMIL':
        continue

    print(f'Processing model: {model_name}')

    # 读取预测结果
    train_df = pd.read_csv(
        train_path,
        dtype={'case_id': str, 'slide_id': str}
    )

    # ===============================
    # 1️⃣ 全部样本
    # ===============================
    cutoff1_all = train_df['risk'].quantile(0.333)
    cutoff2_all = train_df['risk'].quantile(0.667)

    all_results.append({
        'group': 'ALL',
        'model': model_name,
        'cutoff1': cutoff1_all,
        'cutoff2': cutoff2_all
    })

    # ===============================
    # 2️⃣ 各临床亚组
    # ===============================
    for subgroup_col in SUBGROUP_COLUMNS:

        subgroup_df = compute_subgroup_tertile_cutoffs(
            train_df=train_df,
            info_df=info_df,
            subgroup_col=subgroup_col,
            model_name=model_name
        )

        if not subgroup_df.empty:
            all_results.append(subgroup_df)

# ---------- 合并结果 ----------
result_df = pd.concat(
    [r if isinstance(r, pd.DataFrame) else pd.DataFrame([r])
        for r in all_results],
    ignore_index=True
)

# ---------- 保存 ----------
result_df.to_csv(OUTPUT_PATH, index=False)
print(f'\nSaved to: {OUTPUT_PATH}')