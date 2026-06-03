import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # 忽略无关警告


plt.rcParams['font.sans-serif'] = ['SimHei']    # 显示中文（黑体）
plt.rcParams['axes.unicode_minus'] = False      # 显示负号
plt.rcParams['font.size'] = 15                  # 基础字体大小

def plot_contribution_pie(
    excel_path,
    sheet_name,
    save_path=None,
    threshold=None,
    title="各临床因素对DFS/OS预测的相对贡献度",
    figsize=(10, 8),
    dpi=300
):
    plt.rcParams['figure.figsize'] = figsize        # 图表大小

    try:
        # ---------------------- 2. 读取Excel数据 ----------------------
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        # 补充一行Risk
        
        
        # 校验核心列是否存在
        required_cols = ['Covariate', 'Relative_Contribution(%)']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Excel的sheet {sheet_name} 中缺少列：{col}")
        
        # 提取核心数据
        labels = df['Covariate'].tolist()
        sizes = df['Relative_Contribution(%)'].tolist()
        
        # 校验数据有效性
        if len(labels) == 0 or len(sizes) == 0:
            raise ValueError("提取到的贡献度数据为空，请检查Excel中的数据！")
        
        # ---------------------- 3. 可选：合并小占比变量 ----------------------
        if threshold is not None and threshold > 0:
            small_sizes_sum = sum([s for s in sizes if s < threshold])
            new_labels = []
            new_sizes = []
            
            # 保留≥阈值的变量
            for l, s in zip(labels, sizes):
                if s >= threshold:
                    new_labels.append(l)
                    new_sizes.append(s)
            
            # 合并<阈值的变量为"其他"
            if small_sizes_sum > 0:
                new_labels.append("其他")
                new_sizes.append(small_sizes_sum)
            
            labels, sizes = new_labels, new_sizes

        # ---------------------- 4. 绘制饼图（论文风格） ----------------------
        # 配色方案（适配临床论文，可根据需要扩展）
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF', '#CCCCFF', '#FFB366']
        # 突出显示贡献度最大的变量
        explode = [0.1 if s == max(sizes) else 0 for s in sizes]
        
        # 绘制饼图
        patches, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',          # 百分比格式（1位小数）
            explode=explode,            # 突出最大项
            colors=colors[:len(labels)],# 适配变量数量的配色
            shadow=True,                # 阴影增加立体感
            startangle=90,              # 起始角度（正上方向）
            pctdistance=0.85            # 百分比标签位置
        )
        
        # 美化百分比标签
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        # 设置标题
        # plt.title(title, fontsize=14, fontweight='bold', pad=20)
        # 保证饼图为正圆形
        plt.axis('equal')

        # ---------------------- 5. 保存+显示图表 ----------------------
        # 设置默认保存路径
        if save_path is None:
            save_path = excel_path.rsplit('/', 1)[0] + "/临床因素贡献度饼图.png"
        
        # 保存图片（bbox_inches='tight' 防止标题/标签被截断）
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ 饼图已成功保存至：{save_path}")
        
        # 显示图表
        plt.show()
        
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {excel_path}，请检查文件路径是否正确！")
    except KeyError as e:
        print(f"❌ 错误：{e}，请检查sheet名或列名是否正确！")
    except ValueError as e:
        print(f"❌ 错误：{e}")
    except Exception as e:
        print(f"❌ 绘图出错：{str(e)}")
    finally:
        # 清理画布，避免多图绘制时重叠
        plt.clf()
        plt.close()


plot_contribution_pie(
    excel_path="PLOTS/4./各个临床因素的贡献度.xlsx",
    sheet_name="Clin.",  # 替换成你的真实sheet名
    save_path="PLOTS/4./临床因素贡献度饼图.png"
)
    
plot_contribution_pie(
    excel_path="PLOTS/4./各个临床因素的贡献度.xlsx",
    sheet_name="Clin.+Risk",  # 替换成你的真实sheet名
    save_path="PLOTS/4./临床因素贡献度饼图.png"
)