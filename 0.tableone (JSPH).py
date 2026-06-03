import pandas as pd
import glob
from tableone import TableOne
import numpy as np
from scipy.stats import kruskal


def excel_to_custom_dict(excel_path):
    # 1. 读取Excel文件（默认读取第一个工作表，可根据实际情况修改sheet_name）
    df = pd.read_excel(excel_path, sheet_name=0)
    
    # 2. 查看数据结构（可选，用于确认列名，可根据实际Excel列名调整）
    # print("Excel列名：", df.columns.tolist())
    # print("前5行数据：")
    # print(df.head())
    
    # 3. 定义关键列名（需根据你的Excel实际列名调整，此处基于常见TableOne导出格式预设）
    # 若Excel列名不同，需替换为实际列名，例如：变量名列可能是"Variable"，分类标识列可能是"Category"，描述列可能是"Label"
    var_col = df.columns[0]  # 第一列：变量名（如"Age, n (%)"）
    cat_col = df.columns[1]  # 第二列：分类标识（如"0"、"1"）
    desc_col = df.columns[2] # 第三列：分类描述（如"≥65 years"、"<65 years"）
    
    # 4. 数据预处理：向前填充变量名（处理同一变量多行分类的情况）
    df[var_col] = df[var_col].fillna(method='ffill')
    
    # 5. 过滤无效数据（排除空分类标识、空描述的行）
    df_valid = df.dropna(subset=[cat_col, desc_col])
    
    # 6. 构建目标格式字典：key为(变量名, 分类标识字符串)，value为分类描述
    result_dict = {}
    for _, row in df_valid.iterrows():
        # 将分类标识转为字符串（确保key格式统一，如1→"1"）
        cat_key = str(row[cat_col])
        var_key = str(row[var_col])
        result_dict[(var_key, cat_key)] = str(row[desc_col])
    
    return result_dict

def rename_type(row):
    key = (row["Variable"], str(row["type"]))
    return mapping_dict.get(key, row["type"])


def format_tableone(df):
    """
    输入：
        df = 你当前的tableone dataframe（已包含 Variable 和 type 列）

    输出：
        重新排版后的 dataframe
    """

    df = df.copy()

    # ==============================
    # Step 1: 处理列名 (加 n= )
    # ==============================

    # 取第一行 n 那一行
    n_row = df.iloc[0]

    new_columns = []

    for col in df.columns:
        if col in ["Variable", "type", "P-Value"]:
            new_columns.append(col)
        else:
            n_value = n_row[col]
            new_columns.append(f"{col} (n={n_value}), No. (%)")

    df.columns = new_columns

    # 删除第一行 n
    df = df.iloc[1:].reset_index(drop=True)

    # ==============================
    # Step 2: 去掉 ", n (%)"
    # ==============================

    df["Variable"] = df["Variable"].str.replace(", n (%)", "", regex=False)

    # ==============================
    # Step 3: 重构行结构
    # ==============================

    new_rows = []

    grouped = df.groupby("Variable", sort=False)

    for var, group in grouped:

        # ① 插入变量名行
        empty_row = {col: "" for col in df.columns}
        empty_row["Variable"] = var
        new_rows.append(empty_row)

        # ② 插入每个type行
        for _, row in group.iterrows():
            new_rows.append(row.to_dict())

    new_df = pd.DataFrame(new_rows)

    return new_df

def polish_tableone_layout(df):
    """
    调整 tableone 排版：
    1. Variable 每组只保留第一行
    2. P-Value 移动到每组第一行
    """

    df = df.copy()
    new_rows = []

    grouped = df.groupby("Variable", sort=False)

    for var, group in grouped:

        group = group.copy().reset_index(drop=True)

        # 获取该变量的 P-Value（通常在第一行）
        p_value = group["P-Value"].replace("", pd.NA).dropna()
        p_value = p_value.iloc[0] if len(p_value) > 0 else ""

        for i in range(len(group)):

            row = group.loc[i].copy()

            if i == 0:
                # 第一行保留 Variable
                row["Variable"] = var
                row["P-Value"] = p_value
            else:
                # 其余行 Variable 设为空
                row["Variable"] = ""
                row["P-Value"] = ""

            new_rows.append(row)

    new_df = pd.DataFrame(new_rows)

    return new_df

# Histo Type在生存的论文中没有出现
files = glob.glob('ori_files/*/clinical_info_all.csv')
_columns = [
    'cohort', 'case_id','Age', 'Gender', 'CEA', 'CA199', 'Location', 'Stage', 'Pathological T stage', 'Pathological N stage', 'Metastasis', 'Lauren Type',
    'Grade', 'Histo Type', 'Chemotherapy', 'DFS_status', 'OS_status', 'OS', 'DFS'
]

new_df = pd.DataFrame()
for file in files:
    df = pd.read_csv(file, dtype={'case_id': str})
    columns = _columns.copy()
    if 'DFS' in df.columns:
        df.dropna(subset=['OS','DFS'],how='all', inplace=True)
    elif 'PFS' in df.columns:
        df.dropna(subset=['OS','PFS'],how='all', inplace=True)
        columns.remove('DFS')
        columns.append('PFS')
    else:
        print(f'No DFS or PFS in {cohort}')
    df.drop_duplicates(subset=['case_id'], inplace=True)
    cohort = file.split('/')[-2]

    if cohort == 'SXCH':
        fold0_df = pd.read_csv('data_csv_2fold/SXCH/fold0.csv', dtype={'case_id': str})
        train_ids = fold0_df['train'].tolist()
        train_df = df[df['case_id'].isin(train_ids)].copy()
        train_df['cohort'] = cohort + ' Training'
        new_df = pd.concat([new_df, train_df[columns]], axis=0)

        val_ids = fold0_df['val'].tolist()
        val_df = df[df['case_id'].isin(val_ids)].copy()
        val_df['cohort'] = cohort + ' Validation'
        new_df = pd.concat([new_df, val_df[columns]], axis=0)
    else:
        df['cohort'] = cohort
        if not 'DFS_status' in df.columns: # DFS_status替换为PFS_status
            save_columns = columns.copy()
            save_columns.remove('DFS_status')
            save_columns.append('PFS_status')
        else:
            save_columns = columns.copy()
        new_df = pd.concat([new_df, df[save_columns]], axis=0)
new_df.drop_duplicates(subset=['case_id'], inplace=True)
new_df.to_csv('PLOTS/0.tableone/[all_cohort] clinical_info.csv', index=False)

df = pd.read_csv('PLOTS/0.tableone/[all_cohort] clinical_info.csv')
df.drop(columns=['case_id', 'OS', 'DFS', 'PFS'], inplace=True)
df['Age'] = np.where(
    df['Age'].isna(),          # 条件1：如果是NaN
    np.nan,                   # 结果1：保持NaN不变
    np.where(
        df['Age'] > 65,       # 条件2：非空值，判断是否大于65
        1,                    # 结果2：大于65 → 1
        0                     # 结果3：小于等于65 → 0
    )
)
cc = columns.copy()
cc.remove('case_id')
cc.remove('OS')
cc.remove('DFS')
cc.append('PFS_status')
tableone = TableOne(df, columns=cc, groupby='cohort', pval=True,)

tableone.to_excel('PLOTS/0.tableone/[all_cohort] tableone.xlsx')


'美化tableone'
df = pd.read_excel('PLOTS/0.tableone/[all_cohort] tableone.xlsx', header=1)
df.drop(index=[0], inplace=True)
df.drop(columns=['Missing'], inplace=True)
df.rename(columns={'Unnamed: 0': 'Variable','Unnamed: 1': 'type'}, inplace=True)
df["Variable"] = df["Variable"].ffill()

template_path = "PLOTS/0.tableone/template_tableone.xlsx"
template_df = pd.read_excel(template_path)
mapping_dict = excel_to_custom_dict(template_path)
df["type"] = df.apply(rename_type, axis=1)

# 对列JSPH	SXCH	SYSUCC	TCGA_STAD	YYH重新排序
cohort_order = ["SXCH Training", "SXCH Validation", "CMU1H","YYH", "SYSUCC", "TCGA_STAD"]
fixed_cols = ["Variable", "type", "Overall"]
if "P-Value" in df.columns:
    tail_cols = ["P-Value"]
else:
    tail_cols = []
df = df[fixed_cols + cohort_order + tail_cols]
df.to_excel('PLOTS/0.tableone/[all_cohort] tableone_after_format.xlsx', index=False)


'调整整体的行结构'
new_df = format_tableone(df)
# df["Variable"] = df["Variable"].ffill()
new_df = polish_tableone_layout(new_df)


# 算均值和CI

def format_pvalue(p_val):
    if pd.isna(p_val):
        return "-"
    if p_val < 0.001:
        return "<0.001"
    else:
        return f"{p_val:.4f}"

# 【新增】计算多组生存时间的p值
def calc_survival_pvalue(df, cohort_order, col_name, exclude_cohort=None):
    """
    计算多队列生存时间p值
    col_name: OS/DFS/PFS
    exclude_cohort: 需要排除的队列（如JSPH）
    """
    group_data = []
    for cohort in cohort_order:
        # 排除指定队列
        if exclude_cohort and cohort == exclude_cohort:
            continue
        # 提取当前队列的非空数据
        values = df[df["cohort"] == cohort][col_name].dropna()
        if len(values) > 0:
            group_data.append(values)
    
    # 至少2组才能计算p值
    if len(group_data) >= 2:
        h_stat, p_val = kruskal(*group_data)
        return format_pvalue(p_val)
    else:
        return "-"


def calc_median_iqr_table(df, cohort_order):
    """
    计算每个cohort的OS、DFS的median (IQR)，并按指定格式输出
    
    Parameters
    ----------
    df : pd.DataFrame
        包含 OS, DFS, cohort 列的数据
    cohort_order : list
        指定cohort顺序
    
    Returns
    -------
    result_df : pd.DataFrame
        格式化后的结果表
    """
    
    def median_iqr(x):
        x = x.dropna()
        if len(x) == 0:
            return "NA"
        median = np.median(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        return f"{median:.1f} ({q1:.1f}-{q3:.1f})"
    
    # 初始化结果字典
    result = {
        "Variable": ["OS, months", "DFS, months", "PFS, months"],
        "type": ['','',''],
        # "P-Value": [],
    }
    
    # 遍历每个cohort
    for cohort in cohort_order:
        sub_df = df[df["cohort"] == cohort]
        if cohort != 'JSPH':
            dfs_stat = median_iqr(sub_df["DFS"])
            os_stat = median_iqr(sub_df["OS"])
            pfs_stat = '-'
        else:
            dfs_stat = '-'
            os_stat = median_iqr(sub_df["OS"])
            pfs_stat = median_iqr(sub_df["PFS"])

        
        result[cohort] = [os_stat, dfs_stat, pfs_stat]
    
    # 1. OS：所有6个队列计算p值
    os_p = calc_survival_pvalue(df, cohort_order, "OS")
    # 2. DFS：排除JSPH队列，计算剩余5组p值
    dfs_p = calc_survival_pvalue(df, cohort_order, "DFS", exclude_cohort="JSPH")
    # 3. PFS：仅JSPH有数据，无法组间比较，填-
    pfs_p = "-"

    # 填入结果（严格对应Variable顺序）
    result["P-Value"] = [os_p, dfs_p, pfs_p]

    # 转成DataFrame
    result_df = pd.DataFrame(result)
    
    return result_df

import pandas as pd

def merge_tables_by_rows(df1, df2):
    """
    将两个列名不同但语义一致的表按行合并，
    并统一使用 df1 的列名
    
    Parameters
    ----------
    df1 : pd.DataFrame
        目标列名格式（标准格式）
    df2 : pd.DataFrame
        需要对齐并合并的表
    
    Returns
    -------
    merged_df : pd.DataFrame
    """
    
    # Step 1: 去掉可能的多余空格
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Step 2: 建立列名映射（按位置对齐）
    if len(df1.columns) != len(df2.columns):
        raise ValueError("两个df列数不一致，无法直接按位置对齐！")
    
    col_mapping = dict(zip(df2.columns, df1.columns))
    
    # Step 3: 重命名 df2
    df2_renamed = df2.rename(columns=col_mapping)
    
    # Step 4: 保证列顺序一致
    df2_renamed = df2_renamed[df1.columns]
    
    # Step 5: 按行拼接
    merged_df = pd.concat([df1, df2_renamed], axis=0, ignore_index=True)
    
    return merged_df

all_info = pd.read_csv('PLOTS/0.tableone/[all_cohort] clinical_info.csv')
cohort_order = ["SXCH Training", "SXCH Validation", "CMU1H", "YYH", "SYSUCC", "TCGA_STAD"]

result_df = calc_median_iqr_table(all_info, cohort_order)
print(result_df)

a = new_df.drop(new_df.columns[[2]], axis=1)
merged_df = merge_tables_by_rows(a, result_df)

merged_df.to_excel("PLOTS/0.tableone/[all_cohort] tableone_final.xlsx", index=False)

