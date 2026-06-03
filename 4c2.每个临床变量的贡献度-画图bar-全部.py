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
    dpi=300,
    center=None,
):
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
        # if not center is None:
        #     fig.suptitle(center, fontsize=16, fontweight='bold', y=0.92)
        
        # 调整布局（防止标签截断）
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为总标题留出空间

        # ---------------------- 5. 保存+显示 ----------------------
        if save_path is None:
            save_path = excel_path.rsplit('/', 1)[0] + "/双柱状图贡献度对比_viridis.png"
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.savefig(save_path.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
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


from PIL import Image
import math
import matplotlib.pyplot as plt


def merge_saved_figures(
    file_list,
    img_dir,
    save_path,
    ncols=3,
    figsize=(12, 8),
    dpi=300,
    suffix="_双柱状图贡献度对比.png"  # 如果你是svg就改成.svg
):
    """
    将已保存的图拼接成一个panel figure
    
    file_list: ['SXCH-Train', ...]
    img_dir: 图片所在目录
    save_path: 拼接后保存路径
    """

    n = len(file_list)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, center in enumerate(file_list):
        ax = axes[i]

        img_path = f"{img_dir}/{center}{suffix}"

        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(center, fontsize=FONT_SIZE-7)
            ax.axis('off')
        except Exception as e:
            print(f"❌ 读取失败: {img_path}, {e}")
            ax.axis('off')

    # 删除多余子图
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ 拼图完成: {save_path}")

    plt.show()


# ---------------------- 调用示例（直接复制使用） ----------------------
if __name__ == "__main__":
    # 相对贡献度一样，说明和我想的一样，贡献度取决于建模过程
    files = ['SXCH-Train', 'SXCH-Val', 'External', 'YYH', 'SYSUCC', 'TCGA_STAD']
    for center in files:
        plot_double_bar_charts(
            excel_path=f"PLOTS/4.联合建模/各个临床因素贡献度{center}.xlsx",
            save_path=f"PLOTS/4.联合建模/svg/{center}_双柱状图贡献度对比.svg",
            figsize=(8, 8),
            center=center
        )

    merge_saved_figures(
        file_list=files,
        img_dir="PLOTS/4.联合建模/svg",   # 你现在保存的目录
        save_path="PLOTS/4.联合建模/svg/ALL_panel.png",
        ncols=3,
        figsize=(15, 10),
        suffix="_双柱状图贡献度对比.png"   # ⚠️ 注意格式一致
    )