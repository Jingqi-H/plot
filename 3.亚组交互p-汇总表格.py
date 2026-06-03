import pandas as pd
import os
import importlib
import os

def get_group(subgroup):
    subgroup = str(subgroup)
    
    # ---- 特殊规则 ----
    if subgroup.startswith('Stage'):
        return 'Stage'
    elif subgroup.startswith('pT'):
        return 'pT'
    elif subgroup.startswith('pN'):
        return 'pN'
    elif subgroup.startswith('pM'):
        return 'pM'
    
    # ---- 通用规则：取逗号前 ----
    elif ',' in subgroup:
        return subgroup.split(',')[0].strip()
    
    else:
        return subgroup  # fallback（防止异常）
    

ROOT_DIR = 'PLOTS/3.uni_multi_cox/亚组交互p'

writer = pd.ExcelWriter(f'{ROOT_DIR}/subgroup_analysis_summary.xlsx', engine='openpyxl')
for center in ['SXCH-Train', 'SXCH-Val', 'CMU1H','YYH', 'SYSUCC', 'TCGA_STAD', 'External']:
    df = pd.read_excel(f'{ROOT_DIR}/subgroup_analysis_{center}.xlsx')

    df["Main_Group"] = df["Subgroup"].map(get_group)
    df["P_interaction"] = df.groupby("Main_Group")["P_interaction"].transform(
        lambda x: x.where(x.index == x.index[0], "")  # 仅第一行保留值，其余为空
    )
    df_final = df.drop(columns=["Main_Group"])
    
    df_final.to_excel(writer, sheet_name=center, index=False)
writer.close()


