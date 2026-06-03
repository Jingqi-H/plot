import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 全局字体设置
# =========================
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 15

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_cindex_bar(
    model_paths,          # ⭐ dict: {model_name: path}
    color_scheme_path,
    sheet_name="patient_level",
    centers=None,
    save_path=None,
    legend=False
):
    """
    通用C-index柱状图绘制函数（支持多模型自动扩展）
    """

    # =========================
    # 1️⃣ 读取数据
    # =========================
    model_data = {}

    for model_name, path in model_paths.items():
        df = pd.read_excel(path, sheet_name=sheet_name)

        df = (
            df[df["center"].isin(centers)]
            .set_index("center")
            .loc[centers]
        )

        model_data[model_name] = df

    # =========================
    # 2️⃣ 颜色读取
    # =========================
    color_df = pd.read_excel(color_scheme_path)
    color_dict = dict(zip(color_df["Models"], color_df["Colors"]))

    # fallback颜色（防止缺失）
    default_colors = plt.cm.tab10.colors

    # =========================
    # 3️⃣ 提取数值
    # =========================
    cindex_dict = {}
    err_dict = {}

    for i, (model, df) in enumerate(model_data.items()):
        cindex = df["cindex"].values

        err = [
            cindex - df["ci_lower"].values,
            df["ci_upper"].values - cindex
        ]

        cindex_dict[model] = cindex
        err_dict[model] = err

    # =========================
    # 4️⃣ 画图
    # =========================
    n_models = len(model_data)
    x = np.arange(len(centers))

    width = 0.8 / n_models   # ⭐ 自动分配宽度

    fig_width = max(4, len(centers) * 2.5)
    plt.figure(figsize=(fig_width, 5))

    bars_all = []

    colors = plt.cm.Blues(np.linspace(0, 1, len(model_data.keys())+1))
    colors = colors[1:]

    for i, (model, cindex) in enumerate(cindex_dict.items()):
        offset = (i - (n_models - 1) / 2) * width   # ⭐ 居中关键

        # color = color_dict.get(model, default_colors[i % len(default_colors)])
        color = colors[i]
        bars = plt.bar(
            x + offset,
            cindex,
            width,
            yerr=err_dict[model],
            label=model,
            color=color,
        )

        bars_all.append(bars)

    # =========================
    # 5️⃣ 数值标注（自适应）
    # =========================
    y_min, y_max = plt.ylim()
    y_text = y_min + 0.02  # ⭐ 自动靠底
    y_text = 0.46

    

    for i, bars in enumerate(bars_all):
        color = 'white' if i == len(bars_all)-1 else 'black'
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                f"{height:.3f}",
                ha='center',
                va='bottom',
                fontsize=10,
                color=color,
            )

    # =========================
    # 6️⃣ 处理 x 轴标签（不修改原始centers）
    # =========================
    def clean_center_name(name):
        for key in ["-SLIDE", "-DFS", "-PFS"]:
            name = name.replace(key, "")
        return name

    centers_clean = [clean_center_name(c) for c in centers]

    plt.xticks(x, centers_clean)
    plt.ylabel("C-index")
    plt.ylim(0.45, 0.85)

    # plt.legend(frameon=False)
    # legend放在画布内的上面，横着放
    if legend:
        plt.legend(frameon=False, loc="upper right", ncol=len(model_data.keys()))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


moe_result_dir = "final"
_result_dir = "0.0001loadloss"
result_dir = "patient_level_RandomTrainData"

base_path = "PLOTS/1.cindex"

model_paths = {
    "ABMIL": f"{base_path}/[amil_wsi-{result_dir}] cindex_summary.xlsx",
    "DSMIL": f"{base_path}/[dsmil_wsi-{_result_dir}] cindex_summary.xlsx",
    "GLTrans": f"{base_path}/[gltrans_wsi-{_result_dir}] cindex_summary.xlsx",
    "GRASP": f"{base_path}/[moe_wsi-{moe_result_dir}] cindex_summary.xlsx",
}

plot_cindex_bar(
    model_paths,
    color_scheme_path="PLOTS/@source/color scheme (models).xlsx",
    save_path=f"PLOTS/1.cindex/[{moe_result_dir}] cindex_comparison.svg",
    centers = ["SXCH-TRAIN-SLIDE", "SXCH-VAL-SLIDE", "CMU1H","YYH", "SYSUCC", "TCGA-STAD"],
    legend=True,

)

plot_cindex_bar(
    model_paths,
    color_scheme_path="PLOTS/@source/color scheme (models).xlsx",
    save_path=f"PLOTS/1.cindex/[{moe_result_dir}] cindex_comparison-dfs&pfs.svg",
    centers=["SXCH-TRAIN-DFS-SLIDE", "SXCH-VAL-DFS-SLIDE", "CMU1H-DFS","YYH-DFS",'SYSUCC-DFS', 'JSPH-PFS','TCGA-STAD-DFS']
)