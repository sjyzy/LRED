#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flow-guided mask filling with streaming/sliding-window GPU usage.

Key ideas:
  - Do NOT keep all frames/masks/flows on GPU.
  - Each forward/backward step loads only the 2 needed frames (+mask) to GPU,
    computes flow, samples, then frees tensors.
  - Memory ~ O(H*W), independent of #frames.

Requires:
  - torch >= 2.1, torchvision >= 0.16
  - pillow, numpy, tqdm
"""

import argparse
from pathlib import Path
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# ------------- I/O -------------

def load_image_cpu(path): # 加载RGB图像到tensor
    im = Image.open(path).convert('RGB')
    arr = np.array(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    return t

def load_mask_cpu(path, masked_if_nonzero=True):
    m = Image.open(path).convert('L')
    arr = (np.array(m, dtype=np.float32))/255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)    # [1,1,H,W]
    if masked_if_nonzero:
        t = (t > 0.5).float()   # 1 = masked (to fill)
    else:
        t = (t <= 0.5).float()
    return t

def load_flow_cpu(path, device='cuda'):
    # 读取16位tiff，R=X方向，G=Y方向
    im = Image.open(path)
    arr = np.array(im, dtype=np.uint16)  # [H,W,3] 或 [H,W,2]
    if arr.ndim == 3 and arr.shape[2] >= 2:
        fx = arr[...,0].astype(np.float32)
        fy = arr[...,1].astype(np.float32)
    else:
        raise ValueError(f"Unexpected flow format: {arr.shape}")

    # 反归一化: [0,65535] → [-20,20]
    fx = fx / 65535.0 * 40.0 - 20.0
    fy = fy / 65535.0 * 40.0 - 20.0

    flow = np.stack([fx, fy], axis=0)[None]  # [1,2,H,W]
    return torch.from_numpy(flow).to(device)

def save_image(tensor, path):
    t = tensor.clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()
    arr = (t*255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]

# ------------- Geometry -------------

def make_base_coords(h, w, device):
    ys, xs = torch.meshgrid(torch.arange(h, device=device),
                            torch.arange(w, device=device), indexing='ij') # torch.meshgrid表示将两个一维向量生成二维矩阵
    return torch.stack([xs, ys], dim=-1).float().unsqueeze(0)  # [1,H,W,2] 返回一个[1,h,w,2]的矩阵，其中每个元素是[x,y]

def pix2norm(coords, h, w): # 将像素坐标转换为归一化坐标
    x = coords[...,0] # 获取x坐标
    y = coords[...,1] # 获取y坐标
    nx = x / (w-1) * 2 - 1 # 将x坐标映射到[-1,1]之间
    ny = y / (h-1) * 2 - 1 # 将y坐标映射到[-1,1]之间
    return torch.stack([nx, ny], dim=-1) # 将两个坐标合并为一个二维坐标

def sample_bilinear(img, coords_pix): # [1,C,H,W] [1,H,W,2]
    n,c,h,w = img.shape # [1,1,H,W]
    grid = pix2norm(coords_pix, h, w) # 将像素坐标转换为归一化坐标
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True) # 将归一化坐标映射到图像上

def sample_flow(flow, coords_pix):
    n,_,h,w = flow.shape
    grid = pix2norm(coords_pix, h, w)
    f = F.grid_sample(flow, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return f.permute(0,2,3,1)

def pad_to_multiple(img, multiple=8, mode='replicate'):
    n,c,h,w = img.shape
    H = (h + multiple - 1)//multiple*multiple
    W = (w + multiple - 1)//multiple*multiple
    pad = (0, W-w, 0, H-h)
    if pad[1]==0 and pad[3]==0:
        return img, (0,0,0,0)
    return F.pad(img, pad, mode=mode), pad

def unpad(img, pad):
    l,r,t,b = pad
    if (l,r,t,b)==(0,0,0,0): return img
    return img[..., t:img.shape[-2]-b, l:img.shape[-1]-r]

# ------------- RAFT -------------

def load_raft(device='cuda', large=False):
    from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
    if large:
        weights = Raft_Large_Weights.C_T_V2
        model = raft_large(weights=weights).to(device).eval()
    else:
        weights = Raft_Small_Weights.C_T_V2
        model = raft_small(weights=weights).to(device).eval()
    tfm = weights.transforms()
    return model, tfm

@torch.inference_mode()
def compute_flow_pair(model, tfm, img1, img2, device='cuda', amp=True):
    """
    img1,img2: CPU or GPU [1,3,H,W], float[0,1]
    returns flow [1,2,H,W] on device (pixels)
    """
    img1 = img1.to(device, non_blocking=True) # 将图像加载到GPU
    img2 = img2.to(device, non_blocking=True) # 将图像加载到GPU
    img1p, pad = pad_to_multiple(img1, 8, 'replicate') # 将图像填充到8的倍数
    img2p, _   = pad_to_multiple(img2, 8, 'replicate')  # 将图像填充到8的倍数
    i1, i2 = tfm(img1p, img2p) # 对图像进行预处理
    with torch.cuda.amp.autocast(enabled=amp): # 使用自动混合精度
        flows = model(i1, i2) # 计算光流
        flow  = flows[-1] # 获取光流
    flow = unpad(flow, pad) # 将光流还原到原始图像大小
    return flow  # [1,2,H,W] on device 

# ------------- Core (streaming) -------------

@torch.inference_mode()
def fill_one_frame_streaming(
    idx, img_paths, mask_paths, model, tfm, device='cuda', 
    max_search=10, prefer='nearest', masked_if_nonzero=True, amp=True
):
    """
    Streaming forward/backward search without keeping all frames on GPU. 前向或者后向流式搜索填充一帧。
    """
    # Load current frame & mask on CPU, then to GPU once
    img_cur_cpu  = load_image_cpu(img_paths[idx]) # 加载当前帧图像到tensor
    mask_cur_cpu = load_mask_cpu(mask_paths[idx], masked_if_nonzero)  # 加载当前帧掩码到tensor
    H,W = img_cur_cpu.shape[-2:] # 获取当前帧图像的高度和宽度
    img_cur  = img_cur_cpu.to(device, non_blocking=True)  # 将当前帧图像加载到GPU   
    mask_cur = mask_cur_cpu.to(device, non_blocking=True) # 将当前帧掩码加载到GPU

    masked = (mask_cur>0.5)                    # [1,1,H,W] 如果掩码大于0.5，则表示需要填充
    if masked.sum()==0:
        return img_cur                           # no fill needed 

    base_coords = make_base_coords(H, W, device) # [1,H,W,2] 获取基础坐标网格

    # containers
    forward_found = torch.zeros_like(masked, dtype=torch.bool) # [1,1,H,W] 前向搜索是否找到填充
    backward_found= torch.zeros_like(masked, dtype=torch.bool)   # [1,1,H,W] 后向搜索是否找到填充
    forward_color = torch.zeros_like(img_cur) # [1,3,H,W] 前向搜索填充的颜色
    backward_color= torch.zeros_like(img_cur) # [1,3,H,W] 后向搜索填充的颜色
    forward_steps = torch.full_like(masked, 1e9, dtype=torch.float32) # [1,1,H,W] 前向搜索的步数
    backward_steps= torch.full_like(masked, 1e9, dtype=torch.float32) # [1,1,H,W] 后向搜索的步数

    # -------- forward streaming idx -> idx+1 -> ...
    cur_coords = base_coords.clone() # [1,H,W,2] 当前坐标网格
    cur = idx # cur = 0 表示当前帧索引
    for step in range(1, max_search+1): # 前向搜索的最大步数
        if cur >= len(img_paths)-1: break # 如果已经到达最后一帧，则停止前向搜索
        # load just two frames for this step (CPU->GPU)
        img_a_cpu = load_image_cpu(img_paths[cur]) # 加载当前帧图像到tensor
        img_b_cpu = load_image_cpu(img_paths[cur+1]) # 加载下一帧图像到tensor
        # flow = compute_flow_pair(model, tfm, img_a_cpu, img_b_cpu, device=device, amp=amp)  # a->b 获取a到b的flow
        flow_path = str(img_paths[cur+1]).replace('_color.png', '_flow.tiff') # 获取a到b的flow路径
        flow = load_flow_cpu(flow_path, device=device) # 直接加载光流 b->a
        flow = -flow # 取反得到 a->b

        cur_coords = cur_coords + sample_flow(flow, cur_coords)  # 更新当前坐标网格
        cur += 1 # 更新当前帧索引

        # sample target mask & image of cur
        tgt_mask = load_mask_cpu(mask_paths[cur], masked_if_nonzero).to(device, non_blocking=True) # 加载当前帧掩码到tensor
        good = (sample_bilinear(tgt_mask, cur_coords) < 0.3) & (~forward_found) & masked # 判断当前坐标网格是否在掩码内且未在前向搜索中找到

        if good.any(): # 如果有坐标网格在掩码内且未在前向搜索中找到
            img_cur_cpu2 = load_image_cpu(img_paths[cur]) # 加载当前帧图像到tensor
            col = sample_bilinear(img_cur_cpu2.to(device, non_blocking=True), cur_coords) # 根据当前坐标网格采样当前帧图像的颜色
            forward_color = torch.where(good.expand_as(forward_color), col, forward_color) # 更新前向搜索的颜色
            forward_found = torch.where(good, torch.ones_like(forward_found, dtype=torch.bool), forward_found) # 更新前向搜索的找到标志
            forward_steps = torch.where(good, torch.full_like(forward_steps, float(step)), forward_steps) # 更新前向搜索的步数

        # free asap
        del img_a_cpu, img_b_cpu, flow, tgt_mask # 释放内存
        if 'cuda' in str(device): torch.cuda.empty_cache() # 清空cuda缓存

        if (forward_found | (~masked)).all(): # 如果所有坐标网格在前向搜索中找到或未在掩码内
            break

    # -------- backward streaming idx -> idx-1 -> ...
    cur_coords = base_coords.clone() # 初始化当前坐标网格为基准坐标网格
    cur = idx # 初始化当前帧索引为基准帧索引
    for step in range(1, max_search+1): # 从1到max_search进行反向搜索
        if cur <= 0: break # 如果当前帧索引小于等于0，停止搜索
        img_a_cpu = load_image_cpu(img_paths[cur])     # cur # 加载当前帧图像到tensor
        img_b_cpu = load_image_cpu(img_paths[cur-1])   # cur-1 # 加载前一帧图像到tensor
        # flow = compute_flow_pair(model, tfm, img_a_cpu, img_b_cpu, device=device, amp=amp)  # cur->cur-1 # 计算当前帧到前一帧的光流
        flow_path = str(img_paths[cur]).replace('_color.png', '_flow.tiff')
        flow = load_flow_cpu(flow_path)  # cur->cur-1 # 加载当前帧到前一帧的光流

        cur_coords = cur_coords + sample_flow(flow, cur_coords) # 更新当前坐标网格
        cur -= 1 # 更新当前帧索引

        tgt_mask = load_mask_cpu(mask_paths[cur], masked_if_nonzero).to(device, non_blocking=True) # 加载当前帧掩码到tensor
        good = (sample_bilinear(tgt_mask, cur_coords) < 0.3) & (~backward_found) & masked # 判断当前坐标网格是否在掩码内且未在后向搜索中找到

        if good.any(): # 如果有坐标网格满足条件
            img_cur_cpu2 = load_image_cpu(img_paths[cur]) # 加载当前帧图像到tensor
            col = sample_bilinear(img_cur_cpu2.to(device, non_blocking=True), cur_coords) # 在当前坐标网格处采样颜色
            backward_color = torch.where(good.expand_as(backward_color), col, backward_color) # 更新反向颜色
            backward_found = torch.where(good, torch.ones_like(backward_found, dtype=torch.bool), backward_found) # 更新反向搜索标志
            backward_steps = torch.where(good, torch.full_like(backward_steps, float(step)), backward_steps) # 更新反向步数

        del img_a_cpu, img_b_cpu, flow, tgt_mask # 删除中间变量以释放内存
        if 'cuda' in str(device): torch.cuda.empty_cache() # 如果在GPU上运行，则清空缓存

        if (backward_found | (~masked)).all(): # 如果所有坐标网格都已找到或所有坐标网格都在掩码外
            break # 结束循环

    # -------- decide & compose --------
    out = img_cur.clone() # 复制当前帧图像
    use_fwd_only = forward_found & (~backward_found)  # 判断当前坐标网格是否在前向搜索中找到且未在后向搜索中找到
    use_bwd_only = backward_found & (~forward_found)    # 判断当前坐标网格是否在后向搜索中找到且未在前向搜索中找到
    use_both     = forward_found & backward_found # 判断当前坐标网格是否在前向搜索和后向搜索中都找到

    if prefer == 'nearest': # 如果偏好最近的颜色
        if use_both.any(): # 如果在前向搜索和后向搜索中都找到
            fw_nearer = forward_steps < backward_steps # 判断前向步数是否小于后向步数
            out = torch.where((use_both & fw_nearer).expand_as(out), forward_color, out) # 如果前向步数小于后向步数，则使用前向颜色
            out = torch.where((use_both & (~fw_nearer)).expand_as(out), backward_color, out) # 否则使用后向颜色
    else:  # prefer='forward' or 'backward'
        if prefer == 'forward': # 如果偏好前向颜色
            out = torch.where(use_both.expand_as(out), forward_color, out) # 如果在前向搜索和后向搜索中都找到，则使用前向颜色
        else:
            out = torch.where(use_both.expand_as(out), backward_color, out) # 如果在后向搜索和前向搜索中都找到，则使用后向颜色

    out = torch.where(use_fwd_only.expand_as(out), forward_color, out) # 如果当前坐标网格在前向搜索中找到且未在后向搜索中找到，则使用前向颜色
    out = torch.where(use_bwd_only.expand_as(out), backward_color, out) # 如果当前坐标网格在后向搜索中找到且未在前向搜索中找到，则使用后向颜色
    out = torch.where(masked.expand_as(out), out, img_cur)  # 如果当前坐标网格在掩码内，则使用填充颜色，否则使用原始图像颜色
    return out  # on device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, default='/nas_data/ARIS_Public/dataset/DepthEstimation/C3VD/musk/final_musk/cecum_t1_a')
    parser.add_argument('--image_glob', type=str, default='*_color.png')
    parser.add_argument('--mask_glob',  type=str, default='*_color_mask.001.png')
    parser.add_argument('--flow_glob',  type=str, default='*_flow.tiff')
    parser.add_argument('--out_dir',    type=str, default='/nas_data/ARIS_Public/dataset/DepthEstimation/C3VD/musk/inpaint_specular2/cecum_t1_a')
    parser.add_argument('--device',     type=str, default='cuda')
    parser.add_argument('--max_search', type=int, default=100)
    parser.add_argument('--raft_large', action='store_true')
    parser.add_argument('--masked_if_nonzero', action='store_true', default=True,
                        help='mask非零表示需填充(默认)。若你的mask为0=需填充，启动时加 --no-masked_if_nonzero')
    parser.add_argument('--no_amp', action='store_true', help='关闭 AMP，节省算力一致性问题时可用')
    args = parser.parse_args()
    # CUDA_VISIBLE_DEVICES=1 python inpaint_specular4.py
    # 文件列表
    frames_dir = Path(args.frames_dir)
    img_paths  = sorted(frames_dir.glob(args.image_glob), key=natural_key)
    mask_paths = sorted(frames_dir.glob(args.mask_glob),  key=natural_key)
    assert len(img_paths)==len(mask_paths) and len(img_paths)>=2, "图像/掩码数量不匹配或不足2帧"

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    # 读取一帧得分辨率
    H,W = np.array(Image.open(img_paths[0]).convert('RGB')).shape[:2]

    # 模型
    # model, tfm = load_raft(device=device, large=args.raft_large)
    model, tfm = None, None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 主循环：逐帧输出（每帧内部做流式前/后搜索）
    for idx in tqdm(range(len(img_paths)), desc="Filling"):
        filled = fill_one_frame_streaming(
            idx, img_paths, mask_paths, model, tfm, device=device,
            max_search=args.max_search, prefer='nearest',
            masked_if_nonzero=args.masked_if_nonzero, amp=(not args.no_amp)
        )
        save_image(filled, out_dir / img_paths[idx].name)
        mask_img = Image.open(mask_paths[idx])
        mask_img.save(out_dir / mask_paths[idx].name)

    print("Done. Results saved to:", str(out_dir))

if __name__ == '__main__':
    main()