import glob
import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =====================================================
# 1️⃣ 计算某个临床变量的亚组中位风险
# =====================================================
def compute_subgroup_median_cutoff(
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

        cutoff = subgroup_train['risk'].median()

        results.append({
            'group': f'{subgroup_col}_{int(subgroup)}',
            'model': model_name,
            'cutoff': cutoff
        })

    return pd.DataFrame(results)


def feats_stardand(df, featsName=None): 
    df[featsName] = ( df[featsName] + 0.51 ).astype(int) 
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
    df_copy['Lauren Type'] = df_copy['Lauren Type'].map({0:1,1:1,2:2,3:3})

    return df_copy


# ---------- 基础配置 ----------
# RESULT_DIR = '0.0001loadloss'
RESULT_DIR = 'final'
INFO_PATH = 'ori_files/SXCH/clinical_info_all.csv'
# TRAIN_GLOB = f'results/*/{RESULT_DIR}/summary_SXCH-Train_*.csv'
TRAIN_GLOB = f'results/*/{RESULT_DIR}/summary_SXCH-Train_slide_*.csv'
OUTPUT_PATH = f'PLOTS/2.km/{RESULT_DIR}/cutoff_media_risk_score.csv'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

RENAME_MODEL = {
    'amil_wsi': 'AMIL',
    'moe_wsi': 'MoE',
    'dsmil_wsi': 'DSMIL',
    'gltrans_wsi': 'GLTrans',
}

SUBGROUP_COLUMNS = [
    'Age',
    'Gender',
    'CEA',
    'CA199',
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
info_df['Age'] = (info_df['Age'] > 65).astype(int)

missing_counts = info_df.isnull().sum()
columns_to_impute = missing_counts[missing_counts > 0].index.tolist()
remove_cols = ['DFS', 'DFS_status', 'OS', 'OS_status',
            'Microsatellite status', 'HER-2 status', 'PFS', 'PFS_status']
columns_to_impute = [col for col in columns_to_impute if col not in remove_cols]
print('缺失的列:', columns_to_impute)
info_df = impute_missing_values(info_df, columns_to_impute)


# ---------- 查找所有训练结果 ----------
train_paths = glob.glob(TRAIN_GLOB)
print(train_paths)

all_results = []

for train_path in train_paths:

    model_key = train_path.split('/')[1]
    model_name = RENAME_MODEL.get(model_key)

    # 跳过 AMIL
    if model_name != 'MoE':
        continue

    print(f'Processing model: {model_name}')

    # 读取预测结果
    train_df = pd.read_csv(
        train_path,
        dtype={'case_id': str, 'slide_id': str}
    )
    if 'slide' in train_path:
        agg_df = (
            train_df.groupby("case_id")
              .agg({
                  "risk": "mean",
              })
              .reset_index()
        )
        train_df = train_df.drop_duplicates(subset=['case_id']).drop(columns=['risk'])
        train_df = pd.merge(agg_df, train_df, on='case_id', how='left')


    # ===============================
    # 1️⃣ 全部样本
    # ===============================
    cutoff_all = train_df['risk'].median()

    all_results.append({
        'group': 'ALL',
        'model': model_name,
        'cutoff': cutoff_all
    })

    # ===============================
    # 2️⃣ 各临床亚组
    # ===============================
    for subgroup_col in SUBGROUP_COLUMNS:

        subgroup_df = compute_subgroup_median_cutoff(
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


