import openslide
from typing import Tuple, Union
import h5py
from typing import List, Tuple
import os
import openslide
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import pyvips



def extract_wsi_roi(
    slide: openslide.OpenSlide,
    center: Tuple[float, float],
    size: Union[int, Tuple[int, int]],
    *,
    level: int = 0,
    mpp: float = None,
    coord_unit: str = "um",
    size_unit: str = "um",
    auto_clip: bool = True,
    to_rgb: bool = True,
):
    """
    从WSI中提取ROI区域

    Parameters
    ----------
    slide : openslide.OpenSlide
        已加载的WSI对象
    center : (x, y)
        ROI中心坐标（单位由coord_unit决定）
    size : int or (w, h)
        ROI尺寸（单位由size_unit决定）
    level : int, default=0
        提取的金字塔层级
    mpp : float, optional
        微米/像素（µm per pixel），当使用um单位时必须提供
    coord_unit : str, default="um"
        坐标单位: "um" 或 "pixel"
    size_unit : str, default="um"
        尺寸单位: "um" 或 "pixel"
    auto_clip : bool, default=True
        是否自动裁剪越界区域
    to_rgb : bool, default=True
        是否转换为RGB

    Returns
    -------
    roi : PIL.Image
        裁剪得到的ROI图像
    meta : dict
        一些中间信息（方便debug或对齐）
    """

    # ---------------------------
    # 1. 参数规范化
    # ---------------------------
    if isinstance(size, int):
        size = (size, size)

    x, y = center
    w, h = size

    # ---------------------------
    # 2. 单位转换 → level 0 pixel
    # ---------------------------
    if coord_unit == "um":
        assert mpp is not None, "使用um坐标必须提供mpp"
        x = x / mpp
        y = y / mpp

    if size_unit == "um":
        assert mpp is not None, "使用um尺寸必须提供mpp"
        w = w / mpp
        h = h / mpp

    # 转 int
    x, y = int(x), int(y)
    w, h = int(w), int(h)

    # ---------------------------
    # 3. level处理
    # ---------------------------
    downsample = slide.level_downsamples[level]

    w_level = int(w / downsample)
    h_level = int(h / downsample)

    # 注意：location 必须是 level 0 坐标
    x0 = int(x - w / 2)
    y0 = int(y - h / 2)

    # ---------------------------
    # 4. 越界处理
    # ---------------------------
    if auto_clip:
        slide_w, slide_h = slide.dimensions
        x0 = max(0, min(x0, slide_w - w))
        y0 = max(0, min(y0, slide_h - h))

    # ---------------------------
    # 5. 读取ROI
    # ---------------------------
    roi = slide.read_region(
        location=(x0, y0),
        level=level,
        size=(w_level, h_level),
    )

    if to_rgb:
        roi = roi.convert("RGB")

    # ---------------------------
    # 6. 返回meta信息（很有用！）
    # ---------------------------
    meta = {
        "center_level0_px": (x, y),
        "top_left_level0_px": (x0, y0),
        "size_level0_px": (w, h),
        "size_level_px": (w_level, h_level),
        "level": level,
        "downsample": downsample,
    }

    return roi, meta



def cal_scale_bar(
    vis_img: np.ndarray,
    original_mpp: float,
    downsample: float,
    scale_bar_mm: float = 2,
    margin: int = 100,
):
    """
    为缩放后的WSI图像绘制标准物理比例尺（横线+标注）
    自动适配图像下采样倍率，严格对应物理长度
    
    Args:
        blend_img: 输入图像（numpy数组，缩放后的scaled_img）
        original_mpp: WSI原始level0的MPP（微米/像素，必填）
        downsample: 图像的下采样倍率（vis_level对应的缩放倍数，必填）
        scale_bar_mm: 比例尺物理长度，默认2mm
        color: 比例尺颜色，默认黑色
        margin: 比例尺距离右下角的边距
        line_width: 横线粗细
        draw_text: 是否绘制文字标注
    
    Returns:
        绘制好比例尺的PIL图像（可直接save保存）
    """
    # 1. 核心计算：物理长度 → 原始像素 → 缩放后像素
    um_per_mm = 1000
    # 原始图像（level0）中2mm对应的像素长度
    original_px = (scale_bar_mm * um_per_mm) / original_mpp
    # 缩放后图像中，比例尺的像素长度（关键：除以downsample）
    scaled_px = int(original_px / downsample)

    # 2. numpy数组转PIL图像（解决你之前的报错）
    img_w, img_h = vis_img.size

    # # 3. 定位：右下角绘制比例尺（不遮挡病理内容）
    # bar_y = img_h - margin  # 横线Y坐标
    # x_start = img_w - margin - scaled_px  # 横线起点
    # x_end = img_w - margin  # 横线终点
    # 3. 定位：左下角绘制比例尺（不遮挡病理内容）
    bar_y = img_h - margin  # 横线Y坐标
    x_start = margin  # 横线起点
    x_end = margin + scaled_px  # 横线终点

    return x_start, bar_y, x_end, bar_y



def batch_extract_and_visualize_rois(
    slide: openslide.OpenSlide,
    centers: List[Tuple[float, float]],
    size,
    *,
    mpp: float,
    level: int = 0,
    vis_level: int = 3,
    coord_unit: str = "um",
    size_unit: str = "um",
    save_dir: str = None,
    file_name: str = None,
    colors: List[Tuple[int, int, int]] = None,
):
    """
    批量提取ROI，并在WSI缩略图上统一可视化

    Parameters
    ----------
    centers : list of (x, y)
        多个ROI中心
    size : int or (w, h)
        ROI尺寸
    save_dir : str
        ROI patch保存目录
    save_vis_path : str
        可视化图保存路径
    colors : list
        每个ROI的颜色（可选）
    """

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ---------------------------
    # 1. 获取可视化底图
    # ---------------------------
    level_dim = slide.level_dimensions[vis_level]
    vis_img = slide.read_region((0, 0), vis_level, level_dim).convert("RGB")
    draw = ImageDraw.Draw(vis_img)

    downsample = slide.level_downsamples[vis_level]

    # 画scale bar
    x_start, bar_y, x_end, bar_y = cal_scale_bar(
        vis_img=vis_img,
        original_mpp=mpp,
        downsample=downsample,
    )
    draw.line(
        [(x_start, bar_y), (x_end, bar_y)],
        fill="black",
        width=20
    )


    metas = []

    # 默认颜色
    if colors is None:
        colors = [
            (107, 135, 180),
            (143, 192, 113),
            (253, 99, 100),
            (212, 141, 87),
        ]

    # ---------------------------
    # 2. 遍历每个ROI
    # ---------------------------
    for i, center in enumerate(centers):
        # color = colors[i % len(colors)]
        color = '#000000'

        # 👉 调用你已有函数
        roi, meta = extract_wsi_roi(
            slide,
            center=center,
            size=size,
            level=level,
            mpp=mpp,
            coord_unit=coord_unit,
            size_unit=size_unit,
        )

        metas.append(meta)

        # ---------------------------
        # 保存patch
        # ---------------------------
        if save_dir:
            roi.save(os.path.join(save_dir, f"{i+1}.png"))
            roi.close()

        # ---------------------------
        # 坐标映射到 vis_level
        # ---------------------------
        x0, y0 = meta["top_left_level0_px"]
        w, h = meta["size_level0_px"]

        x0_vis = int(x0 / downsample)
        y0_vis = int(y0 / downsample)
        w_vis = int(w / downsample)
        h_vis = int(h / downsample)

        # ---------------------------
        # 画框 + 编号
        # ---------------------------
        draw.rectangle(
            [(x0_vis, y0_vis), (x0_vis + w_vis, y0_vis + h_vis)],
            outline=color,
            width=25,
        )

        draw.text((x0_vis, y0_vis), str(i), fill=color)

    # ---------------------------
    # 3. 保存可视化图
    # ---------------------------
    if save_dir:
        vis_img.save(os.path.join(save_dir, f"roi.png"))

    return metas, vis_img

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores

def get_centers(coord_dir, file_name):
    h5_file = os.path.join(coord_dir, f"{file_name}.h5")
    with h5py.File(h5_file, "r") as h5f:
        coords = h5f["coords"][:]
        ori_scores = h5f['scores'][:]
    
    sorted_coords_list = []
    for expert_id in range(ori_scores.shape[0]):
        if not (expert_id == 0 or expert_id == 2 or expert_id == 4 or expert_id == 5):
            continue

        scores = ori_scores[expert_id]  # 多任务的模型，ori_scores数和任务数一致
        scores = to_percentiles(scores) 
        scores /= 100
        sorted_indices = np.argsort(scores) # 从小到大排序
        sorted_coords = coords[sorted_indices]
        sorted_coords_list.append(sorted_coords)


    centers0 = [i[0].tolist() for i in sorted_coords_list]
    centers1 = [i[1].tolist() for i in sorted_coords_list]
    return centers0, centers1


def drop_roi_in_img(meta, blend_img, downsample, expert_id, save_dir):

    # ---------------------------
    # 坐标映射到 vis_level
    # ---------------------------
    vis_img = blend_img.copy()
    draw = ImageDraw.Draw(vis_img)
    x0, y0 = meta["top_left_level0_px"]
    w, h = meta["size_level0_px"]

    x0_vis = int(x0 / downsample)
    y0_vis = int(y0 / downsample)
    w_vis = int(w / downsample)
    h_vis = int(h / downsample)

    # ---------------------------
    # 画框 + 编号
    # ---------------------------
    color = '#000000'
    draw.rectangle(
        [(x0_vis, y0_vis), (x0_vis + w_vis, y0_vis + h_vis)],
        outline=color,
        width=20,
    )

    vis_img.save(f'{save_dir}/roi_{expert_id}.png')
    print(f'成功保存：{save_dir}/roi_{expert_id}.png')




# exp_id = '0.0001loadloss'
exp_id = 'final'
coord_dir = f'PLOTS/5.热图/{exp_id}/coords_scores'

file_name = '201501316-1,2-HE#1'
centers0 = [
    # (33019, 6615),
    # (3869, 5950),
    # (40655, 13580),
    # (42931, 9872),
    (34352, 7415),
    (39968, 12759),
    (40109, 15091),
    (46232, 4174),

]
coord_unit = "um"
roi_size = 1024

svs_path = f"/media/ubuntu/DATA/SXCH_svs/{file_name.split('#')[0]}.svs"
slide = openslide.OpenSlide(svs_path)
mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
vis_level = 3
down_sample = slide.level_downsamples[vis_level]

'找到ROI，绘制出来，并将ROI在原图中框出来'
save_dir = f"PLOTS/5.热图/{exp_id}/ROIs/{file_name}"
try:
    metas, vis_img = batch_extract_and_visualize_rois(
        slide,
        centers=centers0,
        size=roi_size,
        mpp=mpp,
        level=0,
        vis_level=vis_level,   
        coord_unit=coord_unit,
        size_unit="pix",
        save_dir=save_dir,
        file_name=file_name,
    )
finally:
    slide.close()


'逐个画roi到热图上'
h5_file = f'{coord_dir}/{file_name}.h5'
patch_size = (roi_size,roi_size)
with h5py.File(h5_file, "r") as h5f:
    coords = h5f["coords"][:]
    ori_scores = h5f['scores'][:]

num = 0
for i, expert_id in enumerate([0,2,4,5]):
    # if not expert_id in [0,2,4,5]: # survival的id
    #     continue
    scores = ori_scores[expert_id]  # 多任务的模型，ori_scores数和任务数一致
    meta = metas[i]
    num += 1

    blend_img = Image.open(f'PLOTS/5.热图/{exp_id}/HEATMAP_OUTPUT/expert{expert_id}/slide_level/{file_name}_blockmap.png')
    drop_roi_in_img(meta, blend_img, down_sample, expert_id, save_dir)
    # break
