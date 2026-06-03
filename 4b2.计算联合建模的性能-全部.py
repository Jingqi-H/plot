import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from tqdm import tqdm

np.random.seed(42)
n_bootstrap = 1000

files = ['SXCH-Train', 'SXCH-Val', 'External', 'CMU1H','YYH', 'SYSUCC', 'TCGA_STAD']
for center in files:
    # if center != 'TCGA_STAD':
    #     continue
    
    # =========================
    # 参数设置
    # =========================
    excel_path = f'PLOTS/4.联合建模/cox_results_{center}.xlsx'
    save_csv = f'PLOTS/4.联合建模/cox_cindex_{center}.csv'

    # =========================
    # 读取验证集
    # =========================
    # 如果你的列名不同，需要修改这里
    risk_col = 'risk_score'
    time_col = 'survival_time'
    event_col = 'status'

    # =========================
    # 逐个 sheet 处理
    # =========================
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names

    results = []
    cindex_matrix = {}

    for sheet in sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet)
        
        # 检查 risk_score 列名是否存在
        if risk_col not in df_sheet.columns:
            print(f"Sheet {sheet} 中没有列 {risk_col}, 跳过")
            continue
        
        # 将 risk_score 与验证集 survival 数据合并
        merged_df = df_sheet.copy()
        
        # =========================
        # Bootstrap 计算 C-index
        # =========================
        cindex_list = []
        n_samples = len(merged_df)
        for _ in tqdm(range(n_bootstrap), desc=f'Bootstrap {sheet}'):
            sample_df = merged_df.sample(n=n_samples, replace=True)
            cidx = concordance_index(
                sample_df[time_col],
                -sample_df[risk_col],  # Cox risk score 越大风险越高，所以取负
                sample_df[event_col]
            )
            cindex_list.append(cidx)
        
        cindex_mean = np.mean(cindex_list)
        cindex_lower = np.percentile(cindex_list, 2.5)
        cindex_upper = np.percentile(cindex_list, 97.5)
        
        # 保存小数点后4位
        results.append({
            'Sheet': sheet,
            'C-index_mean': round(cindex_mean, 4),
            'C-index_95%_lower': round(cindex_lower, 4),
            'C-index_95%_upper': round(cindex_upper, 4)
        })

        cindex_matrix[sheet] = cindex_list
        # break

    # =========================
    # 保存结果
    # =========================
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_csv, index=False)
    cindex_df = pd.DataFrame(cindex_matrix)
    cindex_df.to_csv(save_csv.replace('.csv', '_matrix.csv'), index=False)
    print(f"C-index bootstrap 结果已保存至 {save_csv}")
