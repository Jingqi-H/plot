import openslide
from typing import Tuple, Union


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

import openslide
import numpy as np
from PIL import Image, ImageDraw


def visualize_wsi_roi(
    slide: openslide.OpenSlide,
    top_left_level0: tuple,
    size_level0: tuple,
    *,
    vis_level: int = 3,
    save_path: str = None,
    line_width: int = 3,
    color: tuple = (255, 0, 0),
):
    """
    在WSI的低分辨率图（指定level）上绘制ROI框

    Parameters
    ----------
    slide : openslide.OpenSlide
        WSI对象
    top_left_level0 : (x0, y0)
        ROI左上角（level 0坐标）
    size_level0 : (w, h)
        ROI尺寸（level 0像素）
    vis_level : int, default=3
        可视化使用的level（越大越小图）
    save_path : str, optional
        保存路径
    line_width : int, default=3
        框线宽
    color : tuple, default=(255, 0, 0)
        框颜色（RGB）

    Returns
    -------
    vis_img : PIL.Image
        可视化图像
    """

    # ---------------------------
    # 1. 获取level缩略图
    # ---------------------------
    level_dim = slide.level_dimensions[vis_level]
    vis_img = slide.read_region((0, 0), vis_level, level_dim).convert("RGB")

    # ---------------------------
    # 2. 坐标映射（level 0 → vis_level）
    # ---------------------------
    downsample = slide.level_downsamples[vis_level]

    x0, y0 = top_left_level0
    w, h = size_level0

    x0_vis = int(x0 / downsample)
    y0_vis = int(y0 / downsample)

    w_vis = int(w / downsample)
    h_vis = int(h / downsample)

    # ---------------------------
    # 3. 绘制矩形框
    # ---------------------------
    draw = ImageDraw.Draw(vis_img)

    for i in range(line_width):  # 画粗一点的框
        draw.rectangle(
            [
                (x0_vis - i, y0_vis - i),
                (x0_vis + w_vis + i, y0_vis + h_vis + i),
            ],
            outline=color,
        )

    # ---------------------------
    # 4. 保存
    # ---------------------------
    if save_path is not None:
        vis_img.save(save_path)

    return vis_img


slide = openslide.OpenSlide("/media/ubuntu/DATA/SXCH_svs/202109627-1-HE.svs")
mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])


roi, meta = extract_wsi_roi(
    slide,
    center=(10133.34, 7865.95),   # µm
    size=2048,                # µm
    level=0,
    mpp=mpp,
    coord_unit="um",
    size_unit="pix",
)

roi.save("PLOTS/roi.png")
print(meta)


vis_img = visualize_wsi_roi(
    slide,
    top_left_level0=meta["top_left_level0_px"],
    size_level0=meta["size_level0_px"],
    vis_level=3,
    save_path="PLOTS/roi_vis_level3.png",
)