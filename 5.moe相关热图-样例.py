import os
import sys
# 添加项目根目录到Python路径，为了import自己的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import yaml
import argparse
import torch.utils.data as data
import torch.optim as optim
from timeit import default_timer as timer
import pandas as pd
import h5py
import glob
from pathlib import Path
import openslide
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pyvips
from tqdm import tqdm
import shutil
import numpy as np
from PIL import Image, ImageDraw

from main import seed_torch


os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores


def get_seg_mask_custom(scaled_coords, scaled_region_size, scaled_patch_size):
    mask = np.full(np.flip(scaled_region_size), 0).astype(np.uint8)
    for x, y in scaled_coords:
        mask[y:y+scaled_patch_size[1], x:x+scaled_patch_size[0]] = 1

    mask = mask.astype(bool)
    return mask


def rank_scores_draw_tile(scores, coords, svs_path,save_dir,plot_num = 5,vis_level = 0,
                   patch_size = (224,224)):
    '处理score，归一化'
    scores = to_percentiles(scores) 
    scores /= 100

    sorted_indices = np.argsort(scores) # 从小到大排序
    sorted_coords = coords[sorted_indices]

    # # 后20
    # for i in range(plot_num):
    #     coord = sorted_coords[i,:]

    #     slide_odata = openslide.OpenSlide(svs_path)
    #     img_svs = slide_odata.read_region(coord, level=vis_level, 
    #                                     size=patch_size)
    #     img = np.array(img_svs.convert("RGB"))
    #     Image.fromarray(img).save(f'{save_dir}/last_{i}.jpg')
    #     # break

    # 前20
    for i in range(plot_num):
        coord = sorted_coords[-i-1,:]

        slide_odata = openslide.OpenSlide(svs_path)
        img_svs = slide_odata.read_region(coord, level=vis_level, 
                                        size=patch_size)
        img = np.array(img_svs.convert("RGB"))
        Image.fromarray(img).save(f'{save_dir}/top_{i}.jpg')
        # break
    return sorted_coords




def draw_heatmap(scaled_img,scores, scaled_region_size,scaled_coords, scaled_patch_size,tissue_mask, cmap = 'jet'):

    '处理score，归一化'
    scores = to_percentiles(scores) 
    scores /= 100
    threshold = 0.0
    overlap = 0.0
    binarize = False
    blur = True
    alpha = 0.3


    '在coords对应位置，赋值为score（coords没有记录的坐标位置，为0）'
    overlay = np.full(np.flip(scaled_region_size), 0).astype(float)
    counter = np.full(np.flip(scaled_region_size), 0).astype(np.uint16)      
    count = 0

    for idx in range(len(scaled_coords)):
        score = scores[idx]
        coord = scaled_coords[idx]
        if score >= threshold:
            if binarize:
                score=1.0
                count+=1
        else:
            score=0.0
        # accumulate attention
        overlay[coord[1]:coord[1]+scaled_patch_size[1], coord[0]:coord[0]+scaled_patch_size[0]] += score
        # accumulate counter
        counter[coord[1]:coord[1]+scaled_patch_size[1], coord[0]:coord[0]+scaled_patch_size[0]] += 1

    zero_mask = counter == 0
    if binarize:
        overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
    else:
        overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]

    if blur:
        overlay = cv2.GaussianBlur(overlay,tuple((scaled_patch_size * (1-overlap)).astype(int) * 2 +1),0)  
    del counter 


    '将热图之外的区域变成白色，也就是mask是白色'
    threshold = 0.0
    # inferno jet viridis
    cmap = plt.get_cmap(cmap)
    img_cp = scaled_img.copy()
    heat_map_scale = 0.2 # 0.2
    twenty_percent_chunk = max(1, int(len(scaled_coords) * heat_map_scale))
    for idx in range(len(scaled_coords)):  # 一块一块地迭代， 这里绘制的是热图的缩略图，彩色格子特别大，彩色格子没有透明度
        # if (idx + 1) % twenty_percent_chunk == 0:
        #     print('progress: {}/{}'.format(idx, len(scaled_coords)))
        
        score = scores[idx]
        coord = scaled_coords[idx]
        if score >= threshold:

            # attention block
            raw_block = overlay[coord[1]:coord[1]+scaled_patch_size[1], coord[0]:coord[0]+scaled_patch_size[0]]
            
            # image block (either blank canvas or orig image)
            img_block = img_cp[coord[1]:coord[1]+scaled_patch_size[1], coord[0]:coord[0]+scaled_patch_size[0]].copy()

            # color block (cmap applied to attention block)
            color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

            # tissue mask block
            mask_block = tissue_mask[coord[1]:coord[1]+scaled_patch_size[1], coord[0]:coord[0]+scaled_patch_size[0]] 
            # copy over only tissue masked portion of color block
            img_block[mask_block] = color_block[mask_block]

            # rewrite image block
            img_cp[coord[1]:coord[1]+scaled_patch_size[1], coord[0]:coord[0]+scaled_patch_size[0]] = img_block.copy()
    del overlay
    img_heatmap = img_cp.copy()

    '将热图和原图结合起来'
    blend_img = cv2.addWeighted(img_heatmap, alpha, scaled_img, 1 - alpha, 0)
    
    return img_heatmap, blend_img




def draw_scale_bar(
    blend_img: np.ndarray,
    original_mpp: float,
    downsample: float,
    scale_bar_mm: float = 2,
    color: str = "#000000",
    margin: int = 100,
    line_width: int = 20,
    draw_text: bool = True
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
    vis_img = Image.fromarray(blend_img)
    draw = ImageDraw.Draw(vis_img)
    img_w, img_h = vis_img.size

    # 3. 定位：右下角绘制比例尺（不遮挡病理内容）
    bar_y = img_h - margin  # 横线Y坐标
    x_start = img_w - margin - scaled_px  # 横线起点
    x_end = img_w - margin  # 横线终点

    # 4. 绘制比例尺横线
    draw.line(
        [(x_start, bar_y), (x_end, bar_y)],
        fill=color,
        width=line_width
    )

    # 5. 可选：绘制居中文字标注（2mm）
    if draw_text:
        text = f"{scale_bar_mm}mm"
        # 文字居中在横线上方
        text_x = x_start + (scaled_px - len(text) * 30) // 2
        text_y = bar_y - margin // 2
        draw.text((text_x, text_y), text, fill=color)

    return vis_img


def create_heatmaps(target_idx,svs_dir, save_dir, expert_id, vis_level=0, patch_size = (224,224), scale_radio = 16):
    """
    plot_partial_scores: 只保存部分病理图的热图，默认是200个
    """
    save_coords_scores_dir = save_dir + '/coords_scores'
    processing_h5 = glob.glob(save_coords_scores_dir + '/*.h5')
    # save_heatmaps_dir = f'{save_dir}/HEATMAP_OUTPUT_level{vis_level}/expert' + str(expert_id)
    save_heatmaps_dir = f'{save_dir}/HEATMAP_OUTPUT/expert{expert_id}'
    for i in ['slide_level', 'tile_level']:
        os.makedirs(save_heatmaps_dir +'/'+ i, exist_ok=True)
    # print(f'save dir: {save_heatmaps_dir}')

    scale_factor = (1 / scale_radio, 1 / scale_radio)


    for batch_idx, h5_file in tqdm(enumerate(processing_h5), total=len(target_idx), desc="Processing heatmaps"):

        
        file_name = os.path.basename(h5_file).split('.h5')[0]
        if not file_name in target_idx:
            continue
        # if not file_name in ['201501316-1,2-HE#1', '201613820-1,2-HE#1', '201408154-1,2-HE#1', '201712444-1,2-HE#0']:
        L = ['201501316-1,2-HE#1']
        H = ['201712444-1,2-HE#0']
        # LL = ['202101074-4-HE#0']
        # HH = ['201605147-1,2-HE#0']
        if not file_name in L+H:
            continue
        # print(file_name)

        save_file = f'{save_heatmaps_dir}/slide_level/{file_name}_blockmap.png'
        if os.path.exists(save_file):
            continue

        '读取原始svs文件'
        svs_path = f'{svs_dir}/{file_name.split("#")[0]}.svs'
        if not os.path.exists(svs_path):
            print(f'not found svs file: {svs_path}')
            continue

        slide_odata = openslide.OpenSlide(svs_path)
        level_dim = slide_odata.level_dimensions
        down_level = {}
        for i, (w, h) in enumerate(level_dim):
            if i >=1:
                # print(f'scale factor: {w1/w, h1/h}')
                down_level[str(i)] = (w1/w, h1/h)
            else:
                w1 = w
                h1 = h
                down_level['0'] = (1.0,1.0)
        # print(down_level)
        
        '读取每个svs图片对应的patch坐标和分数，由于我保存的坐标是反的，这里要调转过来'
        with h5py.File(h5_file, "r") as h5f:
            coords = h5f["coords"][:]
            ori_scores = h5f['scores'][:]
        scores = ori_scores[expert_id]  # 多任务的模型，ori_scores数和任务数一致

        '获得slide的vis_level的图片'
        # vis_level = 3 # 0
        try:
            image_py = pyvips.Image.openslideload(svs_path,level=vis_level)
            img = np.array(image_py)[...,0:3]
        except:
            img_svs = slide_odata.read_region((0,0), level=vis_level, 
                                            size=level_dim[vis_level])
            img = np.array(img_svs.convert("RGB"))

        '放缩后，可视化原图'
        if vis_level != 0:
            scale_factor = (1 / down_level[str(vis_level)][0], 1 / down_level[str(vis_level)][1])
            scaled_img = img.copy()
        else:
            scaled_img = cv2.resize(img,(0, 0),  fx=scale_factor[1],fy=scale_factor[0],  interpolation=cv2.INTER_AREA)
        w,h = scaled_img.shape[1], scaled_img.shape[0]
        scaled_region_size = (w,h)
        scaled_coords = np.ceil(coords*np.array(scale_factor)).astype(int)
        scaled_patch_size = np.ceil(patch_size*np.array(scale_factor)).astype(int)

        '放缩后，根据坐标获得mask'
        tissue_mask =  get_seg_mask_custom(scaled_coords, scaled_region_size, scaled_patch_size)


        '绘制热图'
        # if not os.path.exists(save_file):
        heat_map, blend_img = draw_heatmap(scaled_img,scores, scaled_region_size,scaled_coords, scaled_patch_size,tissue_mask,cmap = 'jet')
        # Image.fromarray(heat_map).save(save_file.replace('_blockmap.png', '_heatmap.png'))
        # combined_img = np.hstack((scaled_img, blend_img))
        # Image.fromarray(combined_img).save(save_file.replace('_blockmap.png', '_combined.png'))
        '在blend_img里面画scale bar'
        downsample = down_level[str(vis_level)][0]
        original_mpp = float(slide_odata.properties[openslide.PROPERTY_NAME_MPP_X])
        result_img = draw_scale_bar(
            blend_img=blend_img,
            original_mpp=original_mpp,
            downsample=downsample,
            line_width=20  # WSI大图，线条调粗更清晰
        )
        result_img.save(save_file)

        '绘制scores排名前几和后几的patch'
        save_dir = f'{save_heatmaps_dir}/tile_level/{file_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plot_num = int(scores.shape[0]*0.02)
        sorted_coords = rank_scores_draw_tile(scores, coords, svs_path,save_dir,plot_num =plot_num,vis_level=0,
                    patch_size = patch_size)
        
        save_roi_dir = save_dir + '/ROI'
        os.makedirs(save_roi_dir, exist_ok=True)
        save_file = save_roi_dir+'/'
        create_top_roi(scores,coords,scaled_coords,scaled_img,scaled_patch_size,slide_odata,scale_factor,blend_img,save_dir=save_file,plot_num=plot_num,vis_level=vis_level)

        # break


'获得ROI'
def process_roi(slide_odata,coord,scaled_coord, scaled_patch_size, scaled_img,blend_img,save_dir,scale_factor,vis_level=3, rect_size=(256, 256)):
    half_width = rect_size[0] // 2
    half_height = rect_size[1] // 2
    scaled_top_left = (scaled_coord[0] - half_width, scaled_coord[1] - half_height)
    scaled_bottom_right = (scaled_coord[0] + half_width, scaled_coord[1] + half_height)

    # 计算level0的坐标
    rect_size0 = (int(rect_size[0] / scale_factor[0]), int(rect_size[1] / scale_factor[1]))
    half_width = rect_size0[0] // 2
    half_height = rect_size0[1] // 2
    top_left = (coord[0] - half_width, coord[1] - half_height)
    bottom_right = (coord[0] + half_width, coord[1] + half_height)

    # 读取并保存ROI区域
    img_svs = slide_odata.read_region(top_left, level=vis_level, size=rect_size)
    img = np.array(img_svs.convert("RGB"))
    Image.fromarray(img).save(f'{save_dir}_roi_img.jpg')

    # 绘制矩形并保存
    roi_img = scaled_img.copy()
    cv2.rectangle(roi_img, scaled_top_left, scaled_bottom_right, (0, 255, 0), 4)  # 绿色
    cv2.rectangle(roi_img, (scaled_coord[0], scaled_coord[1]), 
                    (scaled_coord[0]+scaled_patch_size[0], scaled_coord[1]+scaled_patch_size[1]), (255, 0, 0), 4)
    cv2.imwrite(f'{save_dir}_rectangle_img.jpg', roi_img)

    cropped_region = blend_img[scaled_top_left[1]:scaled_bottom_right[1], scaled_top_left[0]:scaled_bottom_right[0]]
    Image.fromarray(cropped_region).save(f'{save_dir}_cropped_heatmap.jpg')  # 保存裁剪区域


def create_top_roi(scores,coords,scaled_coords,scaled_img,scaled_patch_size,slide_odata,scale_factor,blend_img,save_dir,plot_num=100,vis_level=3):
    # scale_factor = 
    # scaled_coords = 
    # scaled_img = 
    # scaled_patch_size = 

    scores_norm = to_percentiles(scores) 
    scores_norm /= 100

    sorted_indices = np.argsort(scores_norm) # 从小到大排序
    sorted_scaled_coords = scaled_coords[sorted_indices]
    sorted_coords = coords[sorted_indices]

    # # 后20
    # for i in range(plot_num):
    #     coord = sorted_coords[i,:]
    #     scaled_coord = sorted_scaled_coords[i,:]
    #     coord = sorted_coords[i,:]
    #     process_roi(slide_odata,coord,scaled_coord, scaled_patch_size, scaled_img,blend_img,save_dir+'last'+str(i),scale_factor,vis_level=3, rect_size=(256, 256))

    # 前20：循环调用封装函数
    for i in range(plot_num):
        scaled_coord = sorted_scaled_coords[-i-1,:]
        coord = sorted_coords[-i-1,:]
        process_roi(slide_odata,coord,scaled_coord, scaled_patch_size, scaled_img,blend_img,save_dir+'top'+str(i),scale_factor,vis_level=vis_level, rect_size=(256, 256))

        # break



if __name__=="__main__":

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, 42)
    # exp_id = '0.0001loadloss'
    exp_id = 'final'
    save_dir = f'PLOTS/5.热图/{exp_id}'
    svs_dir = '/media/ubuntu/DATA/SXCH_svs/'

    '获取要画热图的样本 target_idx'
    # sheet_dict = pd.read_excel(
    #     f'PLOTS/5.热图/{exp_id}/SXCH-Val-用于筛选样本.xlsx',
    #     sheet_name=['L', 'H']  # 指定要读取的sheet名称列表
    # )
    # df_L = sheet_dict['L']  # "L" sheet的数据
    # df_H = sheet_dict['H']  # "H" sheet的数据
    # final_df = pd.concat([df_L, df_H], axis=0)
    # target_idx = []
    # for i in os.listdir(svs_dir):
    #     slide_id = i.split('.')[0]
    #     if slide_id in final_df['slide_id'].values:
    #         wax_id = final_df[final_df['slide_id'] == slide_id]['slide_id_wax'].values[0]
    #         target_idx.append(wax_id)
    # print('target_idx:', len(target_idx))


    target_idx = ['201501316-1,2-HE#1', '201712444-1,2-HE#0']
    for expert_id in range(8):
        print('expert id:', expert_id)
        # scale_radio只有在vis_level=1的时候用上
        create_heatmaps(target_idx, svs_dir,save_dir, vis_level=3, expert_id=expert_id,patch_size = (224,224), scale_radio = 16)
        # break

    # temp里的所有数据，按照top0-20的顺序重新整理
    # roi_dir = f'{save_dir}/HEATMAP_OUTPUT/ROI'
    # file_list = glob.glob(f'{roi_dir}/temp/*/*.jpg')
    # for i in file_list:3
    #     dir_ = os.path.basename(i).split("_")[0][4:]
    #     save_dir = f'{roi_dir}/{i.split("/")[-2]}/{dir_}'
    #     os.makedirs(save_dir, exist_ok=True)
    #     shutil.copy(i, save_dir)

    # # 删除file_name/temp里的所有数据
    # shutil.rmtree(f'{roi_dir}/temp')
        