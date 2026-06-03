import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


FONT_SIZE = 18
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = FONT_SIZE


def plot_double_bar_charts(
    excel_path,
    save_path=None,
    figsize=(8, 8),
    dpi=300
):
    """
    绘制两个并排的柱状图：
    - 左子图：sheet "Clin." 的贡献度柱状图
    - 右子图：sheet "Clin.+Risk" 的贡献度柱状图
    - 按贡献度降序排列
    - 使用viridis配色方案
    
    参数说明：
    ----------
    excel_path : str
        Excel文件完整路径（包含Clin.和Clin.+Risk两个sheet）
    save_path : str, 可选
        图片保存路径，默认自动生成
    title : str, 可选
        图表标题
    figsize : tuple, 可选
        图表尺寸（宽×高），默认(16,8)（适配多因素）
    dpi : int, 可选
        保存分辨率，默认300（论文级）
    """
    # ---------------------- 1. 基础样式配置 ----------------------
    plt.rcParams['axes.unicode_minus'] = False      # 负号显示
    # plt.rcParams['figure.figsize'] = figsize        # 图表大小
    # plt.rcParams['font.size'] = 15                  # 基础字体
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
            ax.set_xticklabels(sorted_covariates, rotation=45, ha='center', fontsize=FONT_SIZE-2)
            
            # 设置y轴标签
            ax.set_ylabel('Relative Contribution (%)', fontsize=FONT_SIZE-2)
            
            # 设置子图标题
            ax.set_title(sub_fig_title[i], fontsize=FONT_SIZE, pad=10)
            
            # 添加数值标注
            for i, (bar, val) in enumerate(zip(bars, sorted_contributions)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=FONT_SIZE-7)
            
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
        # fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # 调整布局（防止标签截断）
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为总标题留出空间

        # ---------------------- 5. 保存+显示 ----------------------
        if save_path is None:
            save_path = excel_path.rsplit('/', 1)[0] + "/双柱状图贡献度对比_viridis.png"
        
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

# ---------------------- 调用示例（直接复制使用） ----------------------
if __name__ == "__main__":
    # 替换成你的实际Excel路径
    plot_double_bar_charts(
        excel_path="PLOTS/4.联合建模/各个临床因素的贡献度.xlsx",
        save_path="PLOTS/4.联合建模/双柱状图贡献度对比.svg",
        figsize=(8, 8),
    )