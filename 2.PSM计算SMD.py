import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 15


def compute_smd(df, col, treatment='Chemotherapy'):
    treated = df[df[treatment] == 1][col].dropna()
    control = df[df[treatment] == 0][col].dropna()

    if len(treated) == 0 or len(control) == 0:
        return np.nan

    mean_diff = abs(treated.mean() - control.mean())

    var_t = treated.var(ddof=1)
    var_c = control.var(ddof=1)

    denom = np.sqrt((var_t + var_c) / 2)

    if denom == 0 or np.isnan(denom):
        return np.nan   # ✅ 改这里

    return mean_diff / denom


def prepare_smd_dataframe(df, covariates, treatment='Chemotherapy'):
    df_copy = df[[treatment] + covariates].copy()

    # 自动识别分类变量（object or category）
    categorical_cols = df_copy[covariates].select_dtypes(include=['object', 'category']).columns.tolist()

    # one-hot
    df_encoded = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)

    return df_encoded

def plot(valid_cols_arr, smd_before_arr, smd_after_arr,save_path):

    plt.figure(figsize=(5, len(valid_cols_arr)*0.7))

    y_pos = np.arange(len(valid_cols_arr))

    plt.scatter(smd_before_arr, y_pos, color='#1f77b4', label='Before PSM', s=60)
    plt.scatter(smd_after_arr, y_pos, color='#d62728', label='After PSM', s=60, marker='^') # 设置点是三角形

    plt.plot(smd_before_arr, y_pos, linewidth=1, color='#1f77b4')
    plt.plot(smd_after_arr, y_pos, linewidth=1, color='#d62728')

    # 阈值线（SMD=0.1）
    plt.axvline(x=0.1, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0.0, color='black', linestyle='--', linewidth=1)

    # y轴
    plt.yticks(y_pos, valid_cols_arr)

    # x轴
    plt.xlabel('Standardized Mean Difference')

    # 标题
    plt.title(f'{center} ({sub_group})')

    # 图例，不要边框，放在右上角
    # plt.legend(frameon=False, loc='upper right', fontsize=12)

    # 美化
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # 保存
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

    # print(f'已保存: {save_path}')

# RESULT_DIR = '0.0001loadloss'
RESULT_DIR = 'final'
ROOT_DIR = f'PLOTS/2.km/{RESULT_DIR}/临床&Risk信息'
SAVE_DIR = f'PLOTS/2.km/{RESULT_DIR}/化疗受益PSM-after'
os.makedirs(SAVE_DIR, exist_ok=True)


target_cols = ['Age', 'Gender', 'Location', 'Grade', 'Histo Type',
       'Lauren Type', 'Stage', 'Pathological T stage', 'Pathological N stage', 'Metastasis']

rename_dict = {
        'Age': 'Age',
        'Gender': 'Gender',
        'Location': 'Location',
        'Grade': 'Grade',
        'Histo Type': 'Histo. Type',
        'Lauren Type': 'Lauren Type',
        'Stage': 'Stage',
        'Pathological T stage': 'pT',
        'Pathological N stage': 'pN',
        'Metastasis': 'pM'
    }

for center in ['SXCH-Train+SXCH-Val','SXCH-Train', 'SXCH-Val', 'External', 'External(merged)']: # , 'YYH', 'SYSUCC'
    file_path = f'{ROOT_DIR}/PSM_{center}.xlsx'
    info_path = f'{ROOT_DIR}/{center}_Info.csv'
    if center == 'External(merged)':
        info_path = f'{ROOT_DIR}/External_Info.csv'
    info_df = pd.read_csv(info_path, dtype={'case_id': str})
    info_df = info_df[(info_df['Stage'] != 1)]
    info_df = info_df[(info_df['Stage'] != 4)]
    
    for sub_group in ['All', 'Stage_2', 'Stage_3']:
        print(f'处理{center} ({sub_group})')
        covariates = target_cols.copy()

        # 🔥 before：必须用同一个subgroup的数据
        df_before = info_df.copy()
        if sub_group != 'All':
            sub_g, sub_v = sub_group.split('_')
            sub_v = int(sub_v)
            df_before = df_before[df_before['Stage'] == sub_v]

        psm_df0 = pd.read_excel(file_path, sheet_name=sub_group+'#0', dtype={'Treated_CaseID': str, 'Control_CaseID': str})
        psm_df1 = pd.read_excel(file_path, sheet_name=sub_group+'#1', dtype={'Treated_CaseID': str, 'Control_CaseID': str})
        psm_df = pd.concat([psm_df0, psm_df1], axis=0)

        psm_id = psm_df['Treated_CaseID'].tolist() + psm_df['Control_CaseID'].tolist()
        after_df = info_df[info_df['case_id'].isin(psm_id)]
        print(f'PSM前后的长度:{len(df_before)}->{len(after_df)}')

        # 🔥 one-hot（before & after分别做）
        df_before_encoded = prepare_smd_dataframe(df_before, covariates)
        df_after_encoded = prepare_smd_dataframe(after_df, covariates)

        # common_cols = list(set(df_before_encoded.columns) & set(df_after_encoded.columns))
        # common_cols.remove('Chemotherapy')
        common_cols = covariates.copy()

        smd_before_list = []
        smd_after_list = []
        valid_cols = []

        for col in common_cols:
            smd_before = compute_smd(df_before_encoded, col)
            smd_after = compute_smd(df_after_encoded, col)

            if not np.isnan(smd_before) and not np.isnan(smd_after):
                smd_before_list.append(smd_before)
                smd_after_list.append(smd_after)
                valid_cols.append(col)
        
        valid_cols_arr = [rename_dict.get(c, c) for c in valid_cols]

        
        # if center == 'All_external':
        #     center = 'External'
        save_path = f'{SAVE_DIR}/PSM_{center}_{sub_group}.svg'
        plot(valid_cols_arr, smd_before_list, smd_after_list,save_path)
        break
        
    # break

