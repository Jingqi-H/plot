# CODEX_TASK.md

# GRASP后端分析代码重构任务说明

## 项目定位

本项目已经完成模型训练与风险预测。

本项目不涉及：

* 模型训练
* 模型推理
* 网络结构开发
* 深度学习算法优化

本项目的目标是：

基于模型输出的 risk score 和临床表格，完成统一、规范、可复用的后端分析流程整理。

重点包括：

* 数据整理
* 数据统计
* 风险分组
* 生存分析
* 联合建模
* PSM分析
* 模型性能评估
* 可解释性分析

请优先考虑：

* 代码重构
* 流程统一
* 可维护性

而不是重新实现已有统计方法。

---

# 当前限制（非常重要）

当前项目已经经历多轮开发和迁移。

存在以下情况：

* 大量历史代码
* 多个版本脚本
* 文件命名不统一
* 部分代码已经废弃
* 部分代码是临时实验版本
* 部分代码是论文最终版本

同时：

* 原始数据路径已经发生变化
* 部分硬编码路径已经失效
* 当前代码无法直接运行

因此：

## 第一阶段禁止执行以下操作

不要：

* 运行代码
* 修改路径
* 修复 FileNotFoundError
* 创建虚假测试数据
* 删除代码
* 重写分析逻辑

请仅通过静态代码分析理解项目结构。

即使代码无法运行，也请继续完成项目分析。

---

# 项目数据结构

## 1. Risk Score

Risk Score结果位于：

### 外部验证

@source/results/*/*/results_external/summary_*.csv

### 训练集

@source/results/*/*/summary_SXCH-Train_*.csv

### 内部验证

@source/results/*/*/summary_SXCH-Val_*.csv

规则：

* 文件路径包含 dfs → DFS分析
* 文件路径不包含 dfs → OS分析

主要字段：

* patient_id
* case_id
* slide_id
* slide_id_wax（SXCH存在）
* risk

---

## 2. Clinical Table

各中心临床表格路径：

@source/ori_files/*/clinical_info_all.csv

常见字段：

* case_id
* Age
* Gender
* CEA
* CA199
* Location
* Grade
* Histo Type
* Lauren Type
* Microsatellite status
* HER-2 status
* Chemotherapy
* Stage
* Pathological T stage
* Pathological N stage
* Metastasis
* DFS
* DFS_status
* OS
* OS_status

注意：

不同中心：

* 字段不完全一致
* 有些变量缺失
* 有些中心存在额外变量

分析时需要自动识别并适配。

---

# 最终希望得到的统一分析框架

## Step 1

Data Preparation

输入：

* Risk Score
* Clinical Table

输出：

analysis_table.xlsx

Sheet1：

Full_Table

包含：

* 临床变量
* risk score
* risk group

Sheet2：

Risk_Table

仅包含：

* patient_id
* case_id
* risk

---

## Step 2

Data Statistics

输出：

### TableOne

参考：

0.tableone.py

### 数据统计

统计：

* Patient数量
* Slide数量

生成：

* Table1
* Cohort Summary

---

## Step 3

Risk Stratification

输出：

* cutoff
* risk_group

规则：

* 使用训练集 median risk
* 内部验证使用训练集cutoff
* 外部验证使用训练集cutoff

---

## Step 4

Survival Analysis

基于现有代码进行整理与优化。

不要重新设计统计流程。

### Kaplan-Meier

参考：

* 2.PSM绘制km-before.py
* 2.PSM绘制km-after.py

支持：

* 全队列
* 各中心
* OS
* DFS
* 化疗亚组
* PSM前
* PSM后

图中输出：

* HR
* 95% CI
* Log-rank P

---

### Cox Analysis

参考：

3a.*.py

输出：

#### Univariable Cox

#### Multivariable Cox

表格包含：

* Variable
* HR
* 95% CI
* P value

同时生成：

* Cox Table
* Forest Plot

---

### Subgroup Analysis

参考：

3.*.py

输出表头：

Subgroup

Low Risk

High Risk

HR (95% CI)

P (value)

P

P_interaction (value)

P_interaction

同时支持：

* 表格
* 森林图

---

## Step 5

Clinical Increment Analysis

参考：

### 建模

4b1.联合建模-全部.py

4b1.联合建模-单独对TCGA外推.py

### 可视化

4b3.绘制多列柱状图-全部(以模型为组).py

4b3.绘制多列柱状图-全部(以中心为组).py

比较：

### 单个临床变量

### Clinical-only

### GRASP

### Clinical + GRASP

输出：

* C-index
* Bootstrap CI
* Result Table
* Bar Plot

---

## Step 6

Model Performance Evaluation

参考：

1.绘制性能的柱状图.py

输出：

### OS

### DFS

分别统计：

* C-index
* Bootstrap CI

并绘制性能比较图。

---

## Step 7

PSM Analysis

目前已有：

* PSM前数据
* PSM后数据

希望整理形成统一流程。

输出：

### PSM Matching Table

### PSM Before Analysis

### PSM After Analysis

支持：

* KM
* Cox
* Subgroup

---

## Step 8

Interpretability Analysis

参考：

5.* 开头代码

目前尚未完全整理。

需要支持：

### 专家使用频次统计

参考：

5.moe相关热图-全局.py

输出：

* Expert Usage Frequency

---

### 指定Slide专家热图

参考：

* 5.获取gate_score.py
* 5.moe相关热图-样例.py

输出：

* Expert Heatmap

---

### ROI可视化

参考：

* 5.根据坐标画热图-L.py
* 5.根据坐标画热图-H.py

输出：

* ROI Visualization

---

# 重构原则

优先级：

结果一致性 > 代码美观 > 工程化

任何重构后：

相同输入数据必须得到与原代码一致的：

* Risk Score
* Risk Group
* KM结果
* Cox结果
* C-index结果

如果结果发生变化：

必须明确指出原因。

禁止为了代码整洁而改变统计逻辑。

---

# 可解释性分析说明

当前阶段暂不分析 5.* 相关代码。

原因：

1. 与主要统计分析流程相互独立
2. 运行依赖复杂
3. 路径依赖较多
4. 需要大量WSI资源

因此：

第一阶段请忽略：

- 5.获取gate_score.py
- 5.moe相关热图-全局.py
- 5.moe相关热图-样例.py
- 5.根据坐标画热图-L.py
- 5.根据坐标画热图-H.py

后续将单独作为一个子项目进行整理。

---

# 第一阶段任务（仅分析）

请不要修改代码。

请完成：

## 1

分析项目目录结构

## 2

输出项目结构树

## 3

分析每个脚本功能

例如：

0.tableone.py

→ 数据统计

2.PSM绘制km-before.py

→ KM分析

3a.xxx.py

→ Cox分析

4b.xxx.py

→ 联合建模

5.xxx.py

→ 可解释性分析

---

## 4

梳理脚本依赖关系

分析：

* 输入文件
* 输出文件
* 中间文件

形成数据流图。

---

## 5

识别重复代码

分析：

* 功能重复
* 逻辑重复
* 不同版本实现

---

## 6

识别废弃代码

判断：

* 是否仍被使用
* 是否为历史版本
* 是否可以归档

不要删除。

仅标记。

---

## 7

提出重构方案

包括：

* 推荐目录结构
* 模块划分
* 配置管理方案
* 路径管理方案
* 重构步骤

---

# 第一阶段结束条件

输出：

1. 项目结构分析报告

2. 脚本功能映射表

3. 数据流图

4. 重复代码清单

5. 废弃代码清单

6. 重构方案

完成后停止。

不要修改任何文件。

等待我的确认后再进入第二阶段。
