import pandas as pd
import numpy as np
import glob
from lifelines import CoxPHFitter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


FONT_SIZE = 15
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = FONT_SIZE

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
def run_cox_and_score(train_df, covariate_dict, save_path):
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


def plot_double_bar_charts(
    excel_path,
    save_path=None,
    figsize=(8, 8),
    dpi=300
):
    # ---------------------- 1. 基础样式配置 ----------------------
    plt.rcParams['axes.unicode_minus'] = False      # 负号显示
    plt.rcParams['axes.grid'] = False               # 关闭网格（更简洁）

    try:
        # ---------------------- 2. 读取两个sheet的数据 ----------------------
        # 定义两个目标sheet名
        sheet_names = {
            'Clinical': 2,        # 第一行y值=2
            'Clinical+GRASP': 1    # 第二行y值=1
        }
        
        # 存储所有数据
        all_data = {}
        max_contribution = 0  # 用于统一颜色条范围
        
        for sheet, y_pos in sheet_names.items():
            # 读取sheet数据
            df = pd.read_excel(excel_path, sheet_name=sheet)
            
            # 校验核心列
            required_cols = ['Covariate', 'Relative_Contribution(%)']
            for col in required_cols:
                if col not in df.columns:
                    raise KeyError(f"Sheet [{sheet}] 缺少列：{col}")
            
            # 检查是否有P_value列
            if 'P_value' not in df.columns:
                print(f"警告: Sheet [{sheet}] 缺少P_value列，将使用默认颜色")
                df['P_value'] = 1.0  # 默认值
            
            if 'risk' in df['Covariate'].values:
                # 将risk替换为Risk
                df['Covariate'] = df['Covariate'].str.replace('risk', 'GRASP')
            
            # 按贡献度降序排序
            covariates = df['Covariate'].tolist()
            contributions = df['Relative_Contribution(%)'].tolist()
            p_values = df['P_value'].tolist()
            
            # 校验数据
            if len(covariates) == 0:
                raise ValueError(f"Sheet [{sheet}] 贡献度数据为空！")
            
            # 记录最大贡献度（统一颜色条）
            current_max = max(contributions)
            if current_max > max_contribution:
                max_contribution = current_max
            
            # 存储数据
            all_data[sheet] = {
                'y_pos': y_pos,
                'covariates': covariates,
                'contributions': contributions,
                'p_values': p_values
            }
        
        # ---------------------- 3. 绘制双柱状图 ----------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = {'Clinical': ax1, 'Clinical+GRASP': ax2}
        
        # 遍历两个sheet的数据绘制柱状图
        sub_fig_title = ['Clinical', 'Clinical+GRASP']
        for i, (sheet, data) in enumerate(all_data.items()):
            ax = axes[sheet]
            covariates = data['covariates']
            contributions = data['contributions']
            p_values = data['p_values']
            
            # 按贡献度降序排序
            # sorted_data = sorted(zip(covariates, contributions, p_values), key=lambda x: x[1], reverse=True)
            sorted_data = zip(covariates, contributions, p_values)
            sorted_covariates, sorted_contributions, sorted_p_values = zip(*sorted_data)
            
            # 根据P_value设置颜色：小于0.05为蓝色，否则为橙色
            colors = []
            for p_val in sorted_p_values:
                if p_val < 0.05:
                    colors.append('#5285BD')  # 蓝色
                else:
                    colors.append('#ff7f0e')  # 橙色
            
            # 绘制柱状图
            # 设置柱子宽度
            bar_width = 0.7
            bars = ax.bar(range(len(sorted_covariates)), sorted_contributions, color=colors, alpha=0.9, width=bar_width)
            
            # 设置x轴标签
            ax.set_xticks(range(len(sorted_covariates)))
            ax.set_xticklabels(sorted_covariates, rotation=45, ha='right', fontsize=FONT_SIZE)
            
            # 设置y轴标签
            ax.set_ylabel('Relative Contribution (%)', fontsize=FONT_SIZE)
            
            # 设置子图标题
            ax.set_title(sub_fig_title[i], fontsize=FONT_SIZE, pad=10)
            
            # 添加数值标注
            for i, (bar, val) in enumerate(zip(bars, sorted_contributions)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=FONT_SIZE-3)
            
            # 设置y轴范围
            ax.set_ylim(0, max_contribution * 1.1)
            
            # 美化子图
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines["bottom"].set_linewidth(1.8)
            ax.spines["left"].set_linewidth(1.8)
            ax.grid(axis='y', alpha=0.3)
        
        # ---------------------- 4. 美化图表（论文级） ----------------------
        # 设置总标题
        fig.suptitle('SXCH', fontsize=16, fontweight='bold', y=0.95)
        
        # 调整布局（防止标签截断）
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为总标题留出空间

        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ 双柱状图已保存至：{save_path}")
        plt.show()
        
    except FileNotFoundError:
        print(f"❌ 未找到文件：{excel_path}")
    except KeyError as e:
        print(f"❌ 错误：{e}，请检查sheet名/列名是否正确！")
    except ValueError as e:
        print(f"❌ 错误：{e}")
    except Exception as e:
        print(f"❌ 绘图出错：{str(e)}")
    finally:
        plt.clf()
        plt.close()

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
    'Clinical': ['Age', 'Gender', 'Location', 'Grade', 'HistologicalType', 'LaurenType', 
                 'pT', 'pN', 'pM', ],
    'Clinical+GRASP': ['Age', 'Gender', 'Location', 'Grade', 'HistologicalType', 'LaurenType', 
                 'pT', 'pN', 'pM', 'risk']
}

# ---------- 执行联合建模并保存结果 ----------
save_path = 'PLOTS/4.联合建模/temp.xlsx'
run_cox_and_score(train_cox_df, selected_covariates, save_path)


plot_double_bar_charts(
        excel_path=save_path,
        save_path="PLOTS/4.联合建模/参考论文贡献图.svg",
        figsize=(15, 8),
    )