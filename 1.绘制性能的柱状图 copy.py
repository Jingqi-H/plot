import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 全局字体设置
# =========================
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 15


def plot_cindex_bar(
    amil_path,
    moe_path,
    color_scheme_path,
    sheet_name="patient_level",
    centers=None,
    save_path=None
):
    """
    绘制 AMIL vs MOE 的 C-index 柱状图（含 CI）
    """


    # =========================
    # 1️⃣ 读取数据
    # =========================
    df_amil = pd.read_excel(amil_path, sheet_name=sheet_name)
    df_dsmil = pd.read_excel(moe_path.replace("moe_wsi","dsmil_wsi"), sheet_name=sheet_name)
    df_gltrans = pd.read_excel(moe_path.replace("moe_wsi","gltrans_wsi"), sheet_name=sheet_name)
    df_moe = pd.read_excel(moe_path, sheet_name=sheet_name)

    df_amil = df_amil[df_amil["center"].isin(centers)].set_index("center").loc[centers]
    df_dsmil = df_dsmil[df_dsmil["center"].isin(centers)].set_index("center").loc[centers]
    df_gltrans = df_gltrans[df_gltrans["center"].isin(centers)].set_index("center").loc[centers]
    df_moe = df_moe[df_moe["center"].isin(centers)].set_index("center").loc[centers]

    # =========================
    # 2️⃣ 读取颜色
    # =========================
    color_df = pd.read_excel(color_scheme_path)

    color_dict = dict(zip(color_df["Models"], color_df["Colors"]))

    color_amil = color_dict.get("ABMIL", "#4C72B0")
    color_dsmil = color_dict.get("DSMIL", "#90DA4B")
    color_gltrans = color_dict.get("GLTrans", "#DBDD52")
    color_moe = color_dict.get("MoE", "#DD8452")

    # =========================
    # 3️⃣ 提取数值
    # =========================
    cindex_amil = df_amil["cindex"].values
    cindex_dsmil = df_dsmil["cindex"].values
    cindex_gltrans = df_gltrans["cindex"].values
    cindex_moe = df_moe["cindex"].values

    # 误差线
    err_amil = [
        cindex_amil - df_amil["ci_lower"].values,
        df_amil["ci_upper"].values - cindex_amil
    ]

    err_dsmil = [
        cindex_dsmil - df_dsmil["ci_lower"].values,
        df_dsmil["ci_upper"].values - cindex_dsmil
    ]

    err_gltrans = [
        cindex_gltrans - df_gltrans["ci_lower"].values,
        df_gltrans["ci_upper"].values - cindex_gltrans
    ]

    err_moe = [
        cindex_moe - df_moe["ci_lower"].values,
        df_moe["ci_upper"].values - cindex_moe
    ]

    # =========================
    # 4️⃣ 画图
    # =========================
    x = np.arange(len(centers))
    width = 0.22 # 柱子宽度

    # 自动调整宽度
    width_per_group = 2
    min_width = 4
    fig_width = max(min_width, len(centers) * width_per_group)

    plt.figure(figsize=(fig_width, 5))


    bars1 = plt.bar(
        x - width / 2,
        cindex_amil,
        width,
        yerr=err_amil,
        # capsize=4,
        label="ABMIL",
        color=color_amil
    )

    
    bars2 = plt.bar(
        x + width / 2,
        cindex_dsmil,
        width,
        yerr=err_dsmil,
        # capsize=4,
        label="DSMIL",
        color=color_dsmil
    )
    bars3 = plt.bar(
        x + width / 2 + width,
        cindex_gltrans,
        width,
        yerr=err_gltrans,
        # capsize=4,
        label="GLTrans",
        color=color_gltrans
    )

    bars4 = plt.bar(
        x + width / 2 + width * 2,
        cindex_moe,
        width,
        yerr=err_moe,
        # capsize=4,
        label="GRASP",
        color=color_moe
    )
    # =========================
    # 5️⃣ 数值标注（柱子内部靠底部）
    # =========================
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                0.46,   # 固定在底部略上方
                f"{height*100:.2f}",
                ha='center',
                va='bottom',
                fontsize=10
            )

    # 不要上边界和右边界
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)

    # 重命名centers
    for index, center in enumerate(centers):
        if "SLIDE" in center and "DFS" in center:
            centers[index] = center.replace("-SLIDE", "").replace("-DFS", "")
        elif "SLIDE" in center:
            centers[index] = center.replace("-SLIDE", "")
        elif "PFS" in center:
            centers[index] = center.replace("-PFS", "")
        elif "DFS" in center:
            centers[index] = center.replace("-DFS", "")
        else:
            centers[index] = center

    plt.xticks(x, centers, rotation=0)
    plt.ylabel("C-index")
    plt.ylim(0.45, 0.85)

    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

moe_result_dir = "0.0001loadloss"
result_dir = "patient_level_RandomTrainData"

plot_cindex_bar(
    amil_path=f"PLOTS/1.cindex/[amil_wsi-{result_dir}] cindex_summary.xlsx",
    moe_path=f"PLOTS/1.cindex/[moe_wsi-{moe_result_dir}] cindex_summary.xlsx",
    color_scheme_path="PLOTS/@source/color scheme (models).xlsx",
    save_path=f"PLOTS/1.cindex/[{moe_result_dir}] cindex_comparison.svg",
    centers = ["SXCH-TRAIN-SLIDE", "SXCH-VAL-SLIDE", "YYH", "SYSUCC", "JSPH", "TCGA-STAD"]

)

plot_cindex_bar(
    amil_path=f"PLOTS/1.cindex/[amil_wsi-{result_dir}] cindex_summary.xlsx",
    moe_path=f"PLOTS/1.cindex/[moe_wsi-{moe_result_dir}] cindex_summary.xlsx",
    color_scheme_path="PLOTS/@source/color scheme (models).xlsx",
    save_path=f"PLOTS/1.cindex/[{moe_result_dir}] cindex_comparison-dfs&pfs.svg",
    centers=["SXCH-TRAIN-DFS-SLIDE", "SXCH-VAL-DFS-SLIDE", 'YYH-DFS','SYSUCC-DFS', 'JSPH-PFS','TCGA-STAD-DFS']
)