# Codex 任务说明：后端分析代码整理

## 项目背景

本项目基于模型预测得到的 risk score，以及患者临床表格，进行后端统计分析和可视化。

输入主要包括：

1. risk score 表格，有以下情况：
    - 外部验证：@source/results/*/*/results_external/summary_*.csv
    - 训练：@source/results/*/*/summary_SXCH-Train_*.csv
    - 内部验证：@source/results/*/*/summary_SXCH-Val_*.csv
    - 如果路径有dfs，则这个结果是dfs的，否则是os的
    - 有用的表格信息：
        - patient_id / case_id
        - slide_id（如果有）
        - slide_id_wax（SXCH有）
        - risk

2. 临床表格，
    - 每个中心的路径：@source/ori_files/*/clinical_info_all.csv
    - 表格内容有：case_id,Age,Gender,CEA,CA199,Location,Grade,Histo Type,Lauren Type,Microsatellite status,HER-2 status,Chemotherapy,Stage,Pathological T stage,Pathological N stage,Metastasis,DFS,DFS_status,OS,OS_status
    - 有些中心的临床表格中，有其他临床变量，或者没有这么多变量，需要根据中心不同，选择不同的变量进行分析

## 总体目标


请帮我整理和重构当前项目代码，使其成为一个清晰、可复用、可扩展的后端分析 pipeline。

重点不是重新写模型，而是整理 risk score + clinical table 后续分析流程。

## 需要支持的分析任务

### 1. 数据整理

- 读取 risk score 表格
- 读取 clinical table
- 按 patient_id 合并
- 处理一个 patient 对应多个 slide 的情况
- 支持 patient-level risk score 聚合，例如 mean / max / median
- 检查缺失值
- 输出合并后的 analysis-ready table
- analysis-ready table希望有两个sheet，一个是完整版本的，一个是只有id和risk score的


### 2.数据统计
参考代码：0.开头的

- tableone 表格：0.tableone.py
- 统计case和slide数：你根据tableone表格统计（我好像没有写代码）


### 3. 风险分组

- 支持根据训练集 median risk score 划分 High / Low
- 验证集和外部测试集必须使用训练集 cutoff
- 输出 risk_group

### 4. 生存分析
我现在的代码（主要是2.开头的用于绘制km曲线，3a.开头的用于单多因素分析，4b.开头的做联合建模）， 你在我的代码基础上优化：我之想做PSM，所以我有PSM前后的数据表格用于分析，希望你也有PSM分析的代码

基于我现在的代码，整理并优化：

- Kaplan-Meier 曲线：包括每个中心全部患者的、亚组的、os、dfs、化疗的（2.PSM绘制km-before.py、2.PSM绘制km-after.py），图中有log-rank test和HR
- univariable Cox、multivariable Cox：有表格，和森林图
- 表格除了变量，还要输出 HR, 95% CI, P（表示具体的值）， P value（表示小于0.05还是什么）
- 亚组的交互p（3.开头的代码），表头Subgroup	Low Risk	High Risk	HR(95%CI)	P (val)	P	P_interaction (val)	P_interaction
- 单个临床变量、clinical-only model、GRASP、clinical + GRASP：
    - 建模：4b1.联合建模-全部.py、4b1.联合建模-单独对TCGA外推.py
    - 保存的结果有表格：4b3.绘制多列柱状图-全部(以模型为组).py
    - 保存的结果有柱状图：4b3.绘制多列柱状图-全部(以模型为组).py、4b3.绘制多列柱状图-全部(以中心为组) .py


### 5. 模型性能评估

- C-index、bootstrap 置信区间
- 比较几个模型：1.绘制性能的柱状图.py，DFS和OS分开画

### 6. 可解释性可视化


- 5.开头的代码，我还没完全梳理好
- 几个功能需求：
    - 绘制各个专家的使用频次图：5.moe相关热图-全局.py
    - 绘制指定slide list 的各个专家热图：5.获取gate_score.py、5.moe相关热图-样例.py
    - 绘制指定slide的ROI区域：5.根据坐标画热图-L.py、5.根据坐标画热图-H.py



## 代码整理要求

请先阅读整个项目结构，然后不要马上大规模修改。

请按照以下步骤工作：

1. 总结当前项目文件结构
2. 判断哪些脚本是重复的、临时的、可删除的
3. 提出推荐的新目录结构
4. 给出重构计划
5. 等我确认后，再开始修改代码

## 推荐目录结构

```text
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── example/
├── configs/
│   └── analysis_config.yaml
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── merge_data.py
│   │   └── preprocess.py
│   ├── survival/
│   │   ├── km_analysis.py
│   │   ├── cox_analysis.py
│   │   └── cindex.py
│   ├── statistics/
│   │   ├── bootstrap.py
│   │   └── model_comparison.py
│   ├── visualization/
│   │   ├── plot_km.py
│   │   ├── plot_cindex.py
│   │   └── plot_forest.py
│   └── utils/
│       ├── io.py
│       └── logger.py
├── scripts/
│   ├── 01_prepare_analysis_table.py
│   ├── 02_run_survival_analysis.py
│   ├── 03_compare_models.py
│   └── 04_generate_figures.py
├── results/
│   ├── tables/
│   └── figures/
├── README.md
└── CODEX_TASK.md