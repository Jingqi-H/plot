import pandas as pd
import glob
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from pathlib import Path
import os

# 设置随机种子
np.random.seed(42)

def compute_cindex_ci(
    risk,
    time,
    event,
    n_bootstrap=1000,
    seed=42
):
    risk = np.asarray(risk)
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)

    rng = np.random.default_rng(seed)
    n = len(risk)
    cindex_list = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            c = concordance_index_censored(
                event[idx],
                time[idx],
                risk[idx]
            )[0]
            cindex_list.append(c)
        except Exception:
            continue

    cindex = np.mean(cindex_list)
    ci_lower = np.percentile(cindex_list, 2.5)
    ci_upper = np.percentile(cindex_list, 97.5)

    return round(cindex, 4), round(ci_lower, 4), round(ci_upper, 4), cindex_list

def find_center_name(s):
    sep1 = "summary_"
    sep2 = "_0."

    start_idx = s.find(sep1)
    end_idx = s.find(sep2)

    # 提取中间字符（含异常处理，避免分隔符不存在报错）
    if start_idx != -1 and end_idx != -1 and start_idx + len(sep1) < end_idx:
        result = s[start_idx + len(sep1) : end_idx]
        # 替换下划线为短横线
        result = result.replace('_','-')
        # 字母全都大写
        result = result.upper()
        print("提取结果：", result)
        return result

    else:
        print("错误：字符串中未找到指定分隔符，或分隔符顺序异常",s)
        return None


def evaluate_csv_list(csv_paths, out_excel, agg_func="mean"):
    slide_rows = []
    patient_rows = []

    bootstrap_cindex = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        method_name = Path(csv_path).stem
        center = find_center_name(method_name)

        # time = df["survival_time"].to_numpy()
        # event = (1 - df["censorship"]).to_numpy()
        # risk = df["risk"].to_numpy()

        # -------- slide-level --------
        # c, l, u = compute_cindex_ci(risk, time, event)
        # slide_rows.append({
        #     "center": find_center_name(method_name),
        #     "method": method_name,
        #     "cindex": c,
        #     "ci_lower": l,
        #     "ci_upper": u
        # })

        # -------- patient-level --------
        agg_df = (
            df.groupby("case_id")
              .agg({
                  "risk": agg_func,
                  "survival_time": "first",
                  "censorship": "first"
              })
              .reset_index()
        )

        time_p = agg_df["survival_time"].to_numpy()
        event_p = (1 - agg_df["censorship"]).to_numpy()
        risk_p = agg_df["risk"].to_numpy()

        c, l, u, cindex_list = compute_cindex_ci(risk_p, time_p, event_p)
        patient_rows.append({
            "center": center,
            "method": method_name,
            "cindex": c,
            "ci_lower": l,
            "ci_upper": u
        })
        bootstrap_cindex[center] = cindex_list

    slide_df = pd.DataFrame(slide_rows)
    patient_df = pd.DataFrame(patient_rows)

    bootstrap_cindex_df = pd.DataFrame(bootstrap_cindex)

    with pd.ExcelWriter(out_excel) as writer:
        slide_df.to_excel(writer, sheet_name="slide_level", index=False)
        patient_df.to_excel(writer, sheet_name="patient_level", index=False)
        bootstrap_cindex_df.to_excel(writer, sheet_name="bootstrap_cindex", index=False)


save_dir = 'PLOTS/1.cindex'
os.makedirs(save_dir, exist_ok=True)

print('='*20,'moe_wsi')
result_dir = "final"
moe_reatul_paht = f'results/moe_wsi/{result_dir}'
paths = glob.glob(f'{moe_reatul_paht}/summary_SXCH-*_slide_*.csv') + glob.glob(f'{moe_reatul_paht}/results_external/summary_*.csv')
print(f'Number of paths: {len(paths)}')
# print(paths)
evaluate_csv_list(
    csv_paths=paths,
    out_excel=f"{save_dir}/[{moe_reatul_paht.split('/')[1]}-{moe_reatul_paht.split('/')[2]}] cindex_summary.xlsx",
    agg_func="median"   # or "median"
)

print('='*20,'dsmil_wsi')
result_dir = "0.0001loadloss"
dsmil_reatul_paht = f'results/dsmil_wsi/{result_dir}'
paths = glob.glob(f'{dsmil_reatul_paht}/summary_SXCH-*_slide_*.csv') + glob.glob(f'{dsmil_reatul_paht}/results_external/summary_*.csv')
print(f'Number of paths: {len(paths)}')

evaluate_csv_list(
    csv_paths=paths,
    out_excel=f"{save_dir}/[{dsmil_reatul_paht.split('/')[1]}-{dsmil_reatul_paht.split('/')[2]}] cindex_summary.xlsx",
    agg_func="median"   # or "median"
)

print('='*20,'gltrans_wsi')
result_dir = "0.0001loadloss"
gltrans_reatul_paht = f'results/gltrans_wsi/{result_dir}'
paths = glob.glob(f'{gltrans_reatul_paht}/summary_SXCH-*_slide_*.csv') + glob.glob(f'{gltrans_reatul_paht}/results_external/summary_*.csv')
print(f'Number of paths: {len(paths)}')
evaluate_csv_list(
    csv_paths=paths,
    out_excel=f"{save_dir}/[{gltrans_reatul_paht.split('/')[1]}-{gltrans_reatul_paht.split('/')[2]}] cindex_summary.xlsx",
    agg_func="median"   # or "median"
)

print('='*20,'amil_wsi')
result_dir = "patient_level_RandomTrainData"
reatul_paht = f'results/amil_wsi/{result_dir}'
paths = glob.glob(f'{reatul_paht}/summary_SXCH-*_slide_*.csv') + glob.glob(f'{reatul_paht}/results_external/summary_*.csv')
print(f'Number of paths: {len(paths)}')
evaluate_csv_list(
    csv_paths=paths,
    out_excel=f"{save_dir}/[{reatul_paht.split('/')[1]}-{reatul_paht.split('/')[2]}] cindex_summary.xlsx",
    agg_func="median"   # or "median"
)
