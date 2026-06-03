import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 15


def plot_double_row_scatter(
    excel_path,
    save_path=None,
    title="各临床因素相对贡献度对比（Clin. vs Clin.+Risk）",
    figsize=(16, 8),
    dpi=300
):
    """
    绘制双行水平散点图：
    - 第一行（y=2）：sheet "Clin." 的贡献度结果
    - 第二行（y=1）：sheet "Clin.+Risk" 的贡献度结果
    - 圆圈越大+颜色越深 → 贡献度越大（viridis配色）
    - x轴=临床因素，y轴=分组（Clin./Clin.+Risk）
    
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
    plt.rcParams['font.size'] = 15                  # 基础字体
    plt.rcParams['axes.grid'] = False               # 关闭网格（更简洁）

    try:
        # ---------------------- 2. 读取两个sheet的数据 ----------------------
        # 定义两个目标sheet名
        sheet_names = {
            'Clin.': 2,        # 第一行y值=2
            'Clin.+Risk': 1    # 第二行y值=1
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
            if 'risk' in df['Covariate'].values:
                # 将risk替换为Risk
                df['Covariate'] = df['Covariate'].str.replace('risk', 'Risk')
            
            # 按贡献度降序排序
            # df_sorted = df.sort_values('Relative_Contribution(%)', ascending=False)
            # covariates = df_sorted['Covariate'].tolist()
            # contributions = df_sorted['Relative_Contribution(%)'].tolist()
            covariates = df['Covariate'].tolist()
            contributions = df['Relative_Contribution(%)'].tolist()
            
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
                'contributions': contributions
            }
        
        # ---------------------- 3. 绘制双行散点图 ----------------------
        # fig, ax = plt.subplots()
        # fig, ax = plt.subplots(figsize=(4,6))
        fig, ax = plt.subplots(figsize=(10,3))

        
        # 遍历两个sheet的数据绘制散点
        scatter_collection = []  # 收集散点对象用于统一颜色条
        for sheet, data in all_data.items():
            y_pos = data['y_pos']
            covariates = data['covariates']
            contributions = data['contributions']
            
            # 散点大小：贡献度越大，圆圈越大（限制范围：200-1800）
            scatter_sizes = np.clip(np.array(contributions) * 25 + 200, 200, 1800)
            
            # 绘制散点（核心：viridis配色）
            scatter = ax.scatter(
                x=covariates,                # x轴：临床因素
                y=[y_pos]*len(covariates),   # 固定y值（行）
                c=contributions,             # 颜色映射：贡献度
                s=scatter_sizes,             # 大小映射：贡献度
                cmap='viridis',              # 要求的viridis配色
                vmin=0,                      # 颜色条最小值
                vmax=max_contribution,       # 颜色条最大值（统一）
                alpha=0.9,                   # 透明度
                # edgecolors='darkslategray',  # 边框色（深灰，增加层次感）
                linewidths=1.5,              # 边框宽度
                label=sheet                  # 图例标签
            )
            scatter_collection.append(scatter)
            
            # 添加数值标注（圆圈上方）
            for i, (x, val) in enumerate(zip(covariates, contributions)):
                ax.text(
                    x=x,
                    y=y_pos + 0.2,          # 标注位置
                    # y=y_pos + 0.08,          # 标注位置
                    s=f'{val:.1f}%',         # 保留1位小数
                    ha='center',             # 水平居中
                    va='bottom',             # 垂直靠下
                    fontsize=9,
                    # fontweight='bold',
                    color='black'    # 标注颜色
                    # color='darkslategray'    # 标注颜色
                )
        
        # ---------------------- 4. 美化图表（论文级） ----------------------
        # # 绘制水平线（区分两行）
        # for sheet, y_pos in sheet_names.items():
        #     ax.axhline(y=y_pos, color='lightgray', linestyle='-', linewidth=2, alpha=0.7)
        
        # 设置坐标轴
        # ax.set_xlabel('临床因素', fontsize=12, fontweight='bold')
        # ax.set_ylabel('分组', fontsize=12, fontweight='bold')
        
        # 调整x轴标签（旋转45度避免重叠）
        plt.xticks(rotation=45, ha='right')
        
        # 设置y轴刻度（显示分组名称）
        ax.set_yticks([1, 2])
        ax.set_yticklabels(['Clin.+Risk', 'Clin.'], fontsize=11, fontweight='bold')
        
        # 固定y轴范围
        ax.set_ylim(0.5, 2.5)
        
        # 移除冗余边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('lightgray')
        ax.spines['bottom'].set_color('lightgray')
        
        # 添加统一颜色条（关键：两个sheet共用一个颜色尺度）
        cbar = plt.colorbar(scatter_collection[0], ax=ax, shrink=1)
        # cbar.set_label('Relative variable contribution (%)', fontsize=10, fontweight='bold')
        
        # 设置标题和图例
        # ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        # ax.legend(loc='upper right', frameon=False, fontsize=11)
        
        # 调整布局（防止标签截断）
        plt.tight_layout()

        # ---------------------- 5. 保存+显示 ----------------------
        if save_path is None:
            save_path = excel_path.rsplit('/', 1)[0] + "/双行贡献度散点图_viridis.png"
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ 双行散点图已保存至：{save_path}")
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
    plot_double_row_scatter(
        excel_path="PLOTS/4.联合建模/各个临床因素的贡献度.xlsx",
        save_path="PLOTS/4.联合建模/双行贡献度散点图_viridis.svg",
        title="临床因素贡献度对比（Clin. vs Clin.+Risk）"
    )