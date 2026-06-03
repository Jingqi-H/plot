# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os

# # 设置中文字体和图形样式
# plt.rcParams['font.family'] = ['Arial']
# plt.rcParams['font.size'] = 15

# # 读取CSV文件
# exp_id = 'patient_level_RandomTrainData'
# save_dir = f'PLOTS/5.热图/箱线图'
# os.makedirs(save_dir, exist_ok=True)

# df = pd.read_csv(f'PLOTS/5.热图/{exp_id}_SXCH-Val-用于筛选样本.csv')

# # 查看数据基本信息
# print("数据基本信息:")
# print(f"数据行数: {len(df)}")
# print(f"Grade类别分布:")
# print(df['Grade'].value_counts().sort_index())
# print(f"\nrisk统计信息:")
# print(df['risk'].describe())

# # 数据预处理
# # 处理Grade列中的缺失值（如果有）
# df = df.dropna(subset=['Grade'])

# # 将Grade转换为字符串类型，方便绘图
# if df['Grade'].dtype != 'object':
#     df['Grade'] = df['Grade'].astype(str)

# # 创建图形
# plt.figure(figsize=(5, 8))

# # 使用seaborn绘制箱图
# # 设置箱体宽度
# box_width = 0.5
# ax = sns.boxplot(data=df, x='Grade', y='risk', palette='Set2', width=box_width)

# # 添加点图显示数据分布
# # sns.stripplot(data=df, x='Grade', y='risk', color='black', alpha=0.5, size=3, jitter=True)

# # 隐藏横纵坐标的值
# plt.xticks([])  # 隐藏x轴刻度值
# # plt.yticks([])  # 隐藏y轴刻度值

# # 隐藏坐标轴标签
# plt.xlabel('')
# plt.ylabel('')

# # 添加网格线
# plt.grid(axis='y', alpha=0.3)

# # 自定义legend
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor=sns.color_palette('Set2')[0], label='Well/Moderately'),
#     Patch(facecolor=sns.color_palette('Set2')[1], label='Poorly')
# ]
# plt.legend(handles=legend_elements, loc='upper right', frameon=False)

# # 添加统计信息
# # for i, grade in enumerate(sorted(df['Grade'].unique())):
# #     grade_data = df[df['Grade'] == grade]['risk']
# #     median_val = grade_data.median()
# #     count_val = len(grade_data)
    
# #     # 在箱图上标注中位数和样本数
# #     plt.text(i, median_val + 0.1, f'n={count_val}\nmed={median_val:.2f}', 
# #              ha='center', va='bottom', fontsize=10, fontweight='bold')

# # 美化图形
# plt.tight_layout()

# # 保存图形
# # plt.savefig('5.热图/Grade_risk_distribution.png', dpi=300, bbox_inches='tight')
# plt.savefig(f'{save_dir}/Grade.svg', bbox_inches='tight')

# # 显示图形
# plt.show()

# print(f"\n绘图完成！图形已保存到 {save_dir}/Grade.svg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch

# =====================================================
# 配置参数
# =====================================================

# 实验配置
# EXP_ID = 'patient_level_RandomTrainData'
EXP_ID = '0.0001loadloss'
DATA_FILE = f'PLOTS/5.热图/{EXP_ID}/SXCH-Val-用于筛选样本.csv'
SAVE_DIR = f'PLOTS/5.热图/箱线图'

# 绘图配置
PLOT_CONFIG = {
    'target_column': 'Grade',  # 可以修改为 'Grade', 'Lauren Type', 'Stage' 等
    'legend_labels': ['Well/Moderately', 'Poorly'],  # 图例标签
    'figure_size': (5, 8),     # 图形尺寸
    'box_width': 0.5,          # 箱体宽度
    'show_y_ticks': True,      # 是否显示y轴刻度
    'show_statistics': False,   # 是否显示统计信息
    'font_size': 15,          # 字体大小
    'font_family': 'Arial',   # 字体
}
# PLOT_CONFIG = {
#     'target_column': 'Lauren Type',  
#     'legend_labels': ['Intesrinal','Diffuse', 'Mixed'], 
#     'figure_size': (5, 8),     
#     'box_width': 0.5,          
#     'show_y_ticks': True,     
#     'show_statistics': False,  
#     'font_size': 15,          
#     'font_family': 'Arial',  
# }
# PLOT_CONFIG = {
#     'target_column': 'Stage',  
#     'legend_labels': ['Stage I','Stage II', 'Stage III', 'Stage IV'], 
#     'figure_size': (5, 8),     
#     'box_width': 0.5,          
#     'show_y_ticks': True,     
#     'show_statistics': False,  
#     'font_size': 15,          
#     'font_family': 'Arial',  
# }
# PLOT_CONFIG = {
#     'target_column': 'Histo Type',  
#     'legend_labels': ['Adenocarcinoma','Other'], 
#     'figure_size': (5, 8),     
#     'box_width': 0.5,          
#     'show_y_ticks': True,     
#     'show_statistics': False,  
#     'font_size': 15,          
#     'font_family': 'Arial',  
# }

# =====================================================
# 初始化设置
# =====================================================

def setup_plotting_style():
    """设置绘图样式"""
    plt.rcParams['font.family'] = [PLOT_CONFIG['font_family']]
    plt.rcParams['font.size'] = PLOT_CONFIG['font_size']

def load_and_preprocess_data():
    """加载和预处理数据"""
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(DATA_FILE)
    
    # 数据基本信息
    print("数据基本信息:")
    print(f"数据行数: {len(df)}")
    print(f"目标列 '{PLOT_CONFIG['target_column']}' 分布:")
    print(df[PLOT_CONFIG['target_column']].value_counts().sort_index())
    print(f"\nrisk统计信息:")
    print(df['risk'].describe())
    
    # 数据预处理
    df = df.dropna(subset=[PLOT_CONFIG['target_column']])
    
    # 转换为字符串类型
    if df[PLOT_CONFIG['target_column']].dtype != 'object':
        df[PLOT_CONFIG['target_column']] = df[PLOT_CONFIG['target_column']].astype(str)
    
    return df

# =====================================================
# 统计分析函数
# =====================================================

def perform_statistical_analysis(df):
    """执行统计分析"""
    target_col = PLOT_CONFIG['target_column']
    categories = sorted(df[target_col].unique())
    
    if len(categories) != 2:
        print(f"警告: 目标列 '{target_col}' 需要恰好有2个类别才能进行检验")
        print(f"当前类别数量: {len(categories)}")
        return None
    
    # 获取两个类别的数据
    cat1_data = df[df[target_col] == categories[0]]['risk']
    cat2_data = df[df[target_col] == categories[1]]['risk']
    
    # Mann-Whitney U检验
    u_stat, p_value = mannwhitneyu(cat1_data, cat2_data, alternative='two-sided')
    
    # FDR校正计算q值
    q_value = multipletests([p_value], method='fdr_bh')[1][0]
    
    # 输出结果
    print(f"\n=== {target_col} 统计分析结果 ===")
    print(f"类别 {categories[0]} vs {categories[1]}:")
    print(f"样本数: {len(cat1_data)} vs {len(cat2_data)}")
    print(f"U统计量: {u_stat:.4f}")
    print(f"p值: {p_value:.6f}")
    print(f"q值 (FDR校正): {q_value:.6f}")
    print(f"是否显著 (α=0.05): {'是' if q_value < 0.05 else '否'}")
    
    return {
        'category1': categories[0],
        'category2': categories[1],
        'n1': len(cat1_data),
        'n2': len(cat2_data),
        'u_statistic': u_stat,
        'p_value': p_value,
        'q_value': q_value,
        'significant': q_value < 0.05
    }

# =====================================================
# 绘图函数
# =====================================================

def create_boxplot(df, stat_results=None):
    """创建箱图"""
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])
    
    # 绘制箱图
    sns.boxplot(
        data=df, 
        x=PLOT_CONFIG['target_column'], 
        y='risk', 
        palette='Set2', 
        width=PLOT_CONFIG['box_width'],
        ax=ax
    )
    
    # 隐藏坐标轴
    ax.set_xticks([])  # 隐藏x轴刻度值
    if not PLOT_CONFIG['show_y_ticks']:
        ax.set_yticks([])  # 隐藏y轴刻度值
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # 添加网格线
    ax.grid(axis='y', alpha=0.3)
    
    # 添加图例
    legend_elements = []
    for i in range(len(df[PLOT_CONFIG['target_column']].unique())):
        legend_elements.append(
            Patch(facecolor=sns.color_palette('Set2')[i], label=PLOT_CONFIG['legend_labels'][i])
        )
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)
    
    # 添加统计信息
    if PLOT_CONFIG['show_statistics'] and stat_results:
        q_value = stat_results['q_value']
        
        # 格式化统计文本
        if q_value < 0.0001:
            stat_text = f"Mann-Whitney U test\nq < 0.0001"
        elif q_value < 0.001:
            stat_text = f"Mann-Whitney U test\nq < 0.001"
        elif q_value < 0.01:
            stat_text = f"Mann-Whitney U test\nq < 0.01"
        else:
            stat_text = f"Mann-Whitney U test\nq = {q_value:.4f}"
        
        ax.text(
            0.5, 0.95, stat_text, 
            transform=ax.transAxes, 
            fontsize=12, 
            ha='center', 
            va='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    return fig, ax

# =====================================================
# 保存结果函数
# =====================================================

def save_results(stat_results):
    """保存统计结果"""
    if stat_results:
        stat_df = pd.DataFrame([stat_results])
        stat_file = f"{SAVE_DIR}/{PLOT_CONFIG['target_column']}_statistical_results.csv"
        stat_df.to_csv(stat_file, index=False)
        print(f"统计结果已保存到 {stat_file}")

def save_plot():
    """保存图形"""
    plot_file = f"{SAVE_DIR}/{PLOT_CONFIG['target_column']}.svg"
    plt.savefig(plot_file, bbox_inches='tight')
    print(f"图形已保存到 {plot_file}")

# =====================================================
# 主函数
# =====================================================

def main():
    """主函数"""
    # 初始化
    setup_plotting_style()
    
    # 加载数据
    df = load_and_preprocess_data()
    
    # 统计分析
    stat_results = perform_statistical_analysis(df)
    
    # 创建图形
    fig, ax = create_boxplot(df, stat_results)
    
    # 保存结果
    save_results(stat_results)
    save_plot()
    
    # 显示图形
    plt.show()
    
    print(f"\n绘图完成！")

if __name__ == "__main__":
    main()
