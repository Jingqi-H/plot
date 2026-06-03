import pandas as pd
import matplotlib.pyplot as plt
import os

FONTSIZE = 15
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = FONTSIZE

NUM_PATH = 'PLOTS/0.tableone/各中心的样本数.csv'
plot_case = 'case'

# ---------------------- 3. 读取CSV并绘制饼图 ----------------------
# 读取CSV文件（验证数据，也可直接用上面的csv_df）
plot_df = pd.read_csv(NUM_PATH)



# 准备绘图数据
labels = plot_df['cohort'].tolist()
sizes = plot_df[plot_case+'_num'].tolist()
colors = ['#D4DFE6', '#d6ecfa', '#EDAFB8', '#96CEB4', '#f7e1d7', '#dedbd2', '#b0c4b1']  # 配色（避免单调）

# 绘制饼图
fig, ax = plt.subplots(figsize=(7, 7))  # 设置画布大小
# fig, ax = plt.subplots(figsize=(10, 8))  # 设置画布大小
# 不显示标签
wedges, texts, autotexts = ax.pie(
    sizes,
    # labels=labels,
    colors=colors,
    autopct='%1.1f%%',  # 显示百分比（保留1位小数）
    startangle=90,      # 起始角度（让饼图更美观）
    # textprops={'fontsize': FONTSIZE},  # 标签字体大小
    wedgeprops={'edgecolor': 'white', 'linewidth': 1},  # 饼图边框

)

# 美化百分比文字（比如设为白色，更清晰）
for autotext in autotexts:
    autotext.set_color('black')
    # autotext.set_fontweight('bold')

# 保证饼图为正圆形
ax.axis('equal')

# 保存饼图（高清格式，避免模糊）
pie_chart_path = f'PLOTS/0.tableone/piechart_{plot_case}.svg'
plt.tight_layout()  # 自动调整布局，避免标签重叠
plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
# plt.show()
plt.close()  # 关闭画布释放资源

print(f"饼图已保存至: {pie_chart_path}")