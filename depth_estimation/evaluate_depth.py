# evaluate_c3vd_ppsnet_aligned.py
from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from options2 import MonodepthOptions
from datasets import C3VD_Dataset


from networks.PPSNet import PPSNet_Backbone, PPSNet_Refinement

import optical_flow_funs as OF
from scipy.optimize import leastsq


import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

cv2.setNumThreads(0)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def save_pfm(filename, image, scale=1):
    image = np.flipud(image).astype(np.float32, copy=False)
    if image.ndim == 2:
        color = False
    elif image.ndim == 3 and image.shape[2] in (1, 3):
        color = (image.shape[2] == 3)
    else:
        raise Exception('Image must have HxW or HxWx1/3 dimensions.')
    with open(filename, "wb") as f:
        f.write(('PF\n' if color else 'Pf\n').encode('utf-8'))
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode('utf-8'))
        if image.dtype.byteorder == '<' or (image.dtype.byteorder == '=' and np.little_endian):
            scale = -abs(scale)
        else:
            scale = abs(scale)
        f.write((f"{scale}\n").encode('utf-8'))
        image.tofile(f)

def rel_percent_depth_difference_map(depth_gt, depth_est):
    depth_gt = np.clip(depth_gt, a_min=1e-6, a_max=None)
    return ((depth_gt - depth_est) / depth_gt) * 100.0


def _read_c3vd_scene_list(list_path):
    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"Scene list not found: {list_path}")
    scenes = []
    with open(list_path, "r") as f:
        for line in f:
            scenes.extend([p for p in line.strip().split() if p])
    if not scenes:
        raise ValueError(f"No entries found in list file: {list_path}")
    return scenes

def _build_c3vd_dataloader(opt):
    try_paths = [
        os.path.join(splits_dir, "c3vd", "test_files.txt"),
        os.path.join(splits_dir, "c3vd", "val.txt"),
        os.path.join(splits_dir, opt.eval_split, "test_files.txt"),
        os.path.join(splits_dir, opt.eval_split, "val.txt"),
        '/nas_data/SJY/code/depth_estimation/pplnet/PPSNet-main2/C3VD_splits/val.txt'
    ]
    list_path = next((p for p in try_paths if os.path.isfile(p)), None)
    if list_path is None:
        raise FileNotFoundError("Could not find C3VD scene list under splits/ (tried c3vd/{val,test}.txt and eval_split path)")

    scene_list = _read_c3vd_scene_list(list_path)
    dataset = C3VD_Dataset(data_dir=opt.data_path, list=scene_list, mode='Test')

    return DataLoader(dataset, batch_size=16, shuffle=False,
                      num_workers=opt.num_workers, pin_memory=True, drop_last=False)


def scale_predictions_lmeds(gt_vec, est_vec):
    gt_flat = gt_vec.reshape(-1).astype(np.float64)
    est_flat = est_vec.reshape(-1).astype(np.float64)
    valid = (gt_flat > 0) & (est_flat > 0)
    if not np.any(valid):
        return est_vec
    gt_v, est_v = gt_flat[valid], est_flat[valid]
    def obj(scale):
        s = float(scale[0])
        return np.median((gt_v - s * est_v) ** 2)
    try:
        s_opt = float(leastsq(lambda s: np.array([obj(s)]),
                              np.array([1.0]), maxfev=100)[0][0])
    except Exception:
        s_opt = np.median(gt_v / np.maximum(est_v, 1e-12))
    return est_vec * s_opt


def strip_module(sd):
    return {(k[7:] if isinstance(k, str) and k.startswith("module.") else k): v for k, v in sd.items()}

def load_pps_weights(backbone, refine, opt):
    if getattr(opt, "pps_pretrained_ckpt", None) and os.path.isfile(opt.pps_pretrained_ckpt):
        print(f"-> Loading PPSNet pretrained weights (single ckpt): {opt.pps_pretrained_ckpt}")
        ckpt = torch.load(opt.pps_pretrained_ckpt, map_location="cpu", weights_only=False)
        b_sd = ckpt.get("student_state_dict", ckpt)
        r_sd = ckpt.get("refiner_state_dict", None)
        b_sd = strip_module(b_sd)
        backbone.load_state_dict(b_sd, strict=True)
        if r_sd is not None:
            r_sd = strip_module(r_sd)
            refine.load_state_dict(r_sd, strict=True)
        else:
            print("[warn] ckpt no refiner_state_dict，refine")
        return
    assert getattr(opt, "load_weights_folder", None), "Neither --pps_pretrained_ckpt nor --load_weights_folder provided."
    folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(folder), f"load_weights_folder not found: {folder}"
    b_path = os.path.join(folder, "pps_backbone.pth")
    r_path = os.path.join(folder, "pps_refine.pth")
    assert os.path.isfile(b_path), f"Missing {b_path}"
    assert os.path.isfile(r_path), f"Missing {r_path}"
    print(f"-> Loading Trainer-saved split weights:\n   {b_path}\n   {r_path}")
    b_sd = strip_module(torch.load(b_path, map_location="cpu"))
    r_sd = strip_module(torch.load(r_path, map_location="cpu"))
    missing_b, unexpected_b = backbone.load_state_dict(b_sd, strict=False)
    missing_r, unexpected_r = refine.load_state_dict(r_sd,  strict=False)
    if missing_b or unexpected_b:
        print("[Backbone] missing keys:", missing_b)
        print("[Backbone] unexpected keys:", unexpected_b)
    if missing_r or unexpected_r:
        print("[Refine] missing keys:", missing_r)
        print("[Refine] unexpected keys:", unexpected_r)

# -------------------- 主评估流程 --------------------
def evaluate(opt):
    assert opt.eval_split.lower() == "c3vd", "--eval_split c3vd"

    MIN_DEPTH, MAX_DEPTH = 1e-6, 1.0
    DEPTH_SCALE = 65535.0 * 0.000001525
    EPS = 1e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
    dataloader = _build_c3vd_dataloader(opt)
    dataset = dataloader.dataset 


    print("-> Building PPSNet backbone/refine (vits/14)")
    backbone = PPSNet_Backbone.from_pretrained('LiheYoung/depth_anything_vits14').to(device).eval()
    token_dim = backbone.pretrained.blocks[0].attn.qkv.in_features
    refine = PPSNet_Refinement(disparity_channels=1, encoded_dim=384,
                               feature_dim=token_dim, heads=8).to(device).eval()
    load_pps_weights(backbone, refine, opt)

  
    outputs_root = None
    if hasattr(opt, "log_dir") and opt.log_dir:
        outputs_root = os.path.join(opt.log_dir, "pps_eval_outputs")
    elif getattr(opt, "load_weights_folder", None):
        outputs_root = os.path.join(opt.load_weights_folder, "pps_eval_outputs")
    else:
        outputs_root = os.path.join(".", "pps_eval_outputs")
    os.makedirs(outputs_root, exist_ok=True)
    img_dir   = os.path.join(outputs_root, "images")
    gt_dir    = os.path.join(outputs_root, "gt")
    pred_dir  = os.path.join(outputs_root, "results")
    errm_dir  = os.path.join(outputs_root, "depth_error_maps")
    for d in [img_dir, gt_dir, pred_dir, errm_dir]:
        os.makedirs(d, exist_ok=True)


    print("-> Running inference (518x518, ppstest-aligned)...")
    pred_disps = []

    def _normalize_light_data(batch_light, B, dtype, device_):
        light_pos, light_dir, mu = batch_light
        def to_tensor(x): return x if torch.is_tensor(x) else torch.as_tensor(x)
        light_pos, light_dir, mu = to_tensor(light_pos), to_tensor(light_dir), to_tensor(mu)
        if isinstance(light_pos, (list, tuple)): light_pos = torch.stack([to_tensor(t) for t in light_pos], 0)
        if isinstance(light_dir, (list, tuple)): light_dir = torch.stack([to_tensor(t) for t in light_dir], 0)
        if isinstance(mu, (list, tuple)):        mu        = torch.stack([to_tensor(t) for t in mu],        0)
        light_pos = light_pos.to(device=device_, dtype=dtype)
        light_dir = light_dir.to(device=device_, dtype=dtype)
        mu        = mu.to(device=device_, dtype=dtype)
        if light_pos.dim()==1: light_pos = light_pos.unsqueeze(0).expand(B, -1)
        if light_dir.dim()==1: light_dir = light_dir.unsqueeze(0).expand(B, -1)
        if mu.dim()==0:        mu = mu.expand(B).clone()
        elif mu.dim()==1 and mu.shape[0]==1: mu = mu.expand(B).clone()
        return (light_pos, light_dir, mu)

    H_infer = W_infer = 518

    with torch.no_grad():
        for data in dataloader:
            img = data["image"].to(device)
            B, _, H0, W0 = img.shape

            img_518 = F.interpolate(img, size=(H_infer, W_infer), mode="bicubic", align_corners=False)

            K_pix = torch.as_tensor(data["intrinsics"]).to(device).float()
            if K_pix.dim() == 2:
                K_pix = K_pix.unsqueeze(0).repeat(B, 1, 1)
            scale_x = W_infer / float(W0)
            scale_y = H_infer / float(H0)
            K_518 = K_pix.clone()
            K_518[:, 0, 0] *= scale_x
            K_518[:, 1, 1] *= scale_y
            K_518[:, 0, 2] = K_pix[:, 0, 2] * scale_x
            K_518[:, 1, 2] = K_pix[:, 1, 2] * scale_y

            M = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
            M[:, 0, 0] = 2.0 / W_infer
            M[:, 0, 2] = 1.0 / W_infer - 1.0
            M[:, 1, 1] = 2.0 / H_infer
            M[:, 1, 2] = 1.0 / H_infer - 1.0
            nK_518 = torch.bmm(M, K_518)

            ref_dirs = OF.get_camera_pixel_directions((H_infer, W_infer), nK_518,
                                                      normalized_intrinsics=True).to(device)
            light_data = _normalize_light_data(data["light_data"], B=B, dtype=img_518.dtype, device_=device)

            disp_coarse, rgb_feats, colored_feats = backbone(
                img_518, ref_dirs, *light_data, nK_518
            )
            disp_refined = refine(rgb_feats, colored_feats, disp_coarse)
            disp = disp_refined.unsqueeze(1)

            pred_disps.append(disp.cpu().numpy()[:, 0])

    pred_disps = np.concatenate(pred_disps, axis=0)

  
    if opt.save_pred_disps:
        out_path = os.path.join(opt.load_weights_folder or ".", f"disps_{opt.eval_split}.npy")
        print("-> Saving predicted disparities to", out_path)
        np.save(out_path, pred_disps)

   
    print("-> PPSNet-style evaluation + saving outputs (preserve original folder structure)")
    abs_rel_list, sq_rel_list, rmse_list, rmse_log_list, a1_list = [], [], [], [], []

  
    if hasattr(dataset, "images"):
        filenames = dataset.images
    else:
       
        filenames = getattr(dataset, "image_paths", None) or getattr(dataset, "files", None)
        if filenames is None:
            raise RuntimeError("Dataset does not expose 'images' list; please modify C3VD_Dataset to keep list in .images")

   
    dataloader = _build_c3vd_dataloader(opt)

    H_eval = W_eval = 384
    ptr = 0
    global_idx = 0  
    for data in dataloader:
        B = data['image'].shape[0]
        disps_b = pred_disps[ptr:ptr+B]; ptr += B
        imgs_b = data['image'].numpy()

        for b in range(B):
           
            img_path = filenames[global_idx]
            global_idx += 1

          
            try:
                rel_path = os.path.relpath(img_path, opt.data_path)
            except Exception:
                
                rel_path = os.path.basename(os.path.dirname(img_path)) + os.path.sep + os.path.basename(img_path)

            rel_dir = os.path.dirname(rel_path)  
            base_name = os.path.splitext(os.path.basename(img_path))[0]

        
            d_img  = os.path.join(img_dir,  rel_dir);  os.makedirs(d_img,  exist_ok=True)
            d_gt   = os.path.join(gt_dir,   rel_dir);  os.makedirs(d_gt,   exist_ok=True)
            d_pred = os.path.join(pred_dir, rel_dir);  os.makedirs(d_pred, exist_ok=True)
            d_err  = os.path.join(errm_dir,  rel_dir);  os.makedirs(d_err,  exist_ok=True)

            gt_depth = data['depth'][b, 0].numpy()
            mask_in  = data['mask'][b, 0].numpy()
            pred_disp = disps_b[b]

            pred_depth = 1.0 / np.maximum(pred_disp, 1e-12)
            pred_depth = np.clip(pred_depth, 0.0, 1.0)
            dmax = max(pred_depth.max(), 1e-12)
            pred_depth = pred_depth / dmax

            gt_resized   = cv2.resize(gt_depth,   (W_eval, H_eval), interpolation=cv2.INTER_CUBIC)
            mask_resized = cv2.resize(mask_in,    (W_eval, H_eval), interpolation=cv2.INTER_NEAREST)
            pred_resized = cv2.resize(pred_depth, (W_eval, H_eval), interpolation=cv2.INTER_CUBIC)

            valid = (mask_resized > 0.5) & (gt_resized > MIN_DEPTH) & (gt_resized < MAX_DEPTH)

    
            img_vis = imgs_b[b] 
            img_vis = np.transpose(img_vis, (1, 2, 0)).astype(np.float32)
            img_vis_384 = cv2.resize(img_vis, (W_eval, H_eval), interpolation=cv2.INTER_CUBIC)
            img_vis_384 = np.clip(img_vis_384, 0.0, 1.0)
            plt.imsave(os.path.join(d_img, f"{base_name}.png"), img_vis_384)

       
            gt_m_full   = gt_resized * DEPTH_SCALE
            pred_m_full = pred_resized * DEPTH_SCALE
            if np.any(valid):
                pred_m_scaled = pred_m_full.copy()
                pred_m_scaled[valid] = scale_predictions_lmeds(gt_m_full[valid], pred_m_full[valid])
            else:
                pred_m_scaled = pred_m_full

            if np.any(valid):
                eps = EPS
                gt_v   = gt_m_full[valid]
                pred_v = pred_m_scaled[valid]
                abs_rel = np.mean(np.abs(gt_v - pred_v) / (gt_v + eps))
                sq_rel  = np.mean(((gt_v - pred_v) ** 2) / (gt_v + eps))
                rmse    = np.sqrt(np.mean((gt_v - pred_v) ** 2))
                rmse_log= np.sqrt(np.mean((np.log(gt_v + eps) - np.log(pred_v + eps)) ** 2))
                a1      = np.mean((np.abs(gt_v - pred_v) / (gt_v + eps)) < 0.1).astype(np.float32)

                abs_rel_list.append(abs_rel); sq_rel_list.append(sq_rel)
                rmse_list.append(rmse); rmse_log_list.append(rmse_log); a1_list.append(a1)


            save_pfm(os.path.join(d_gt,   f"{base_name}.pfm"),   gt_m_full.astype(np.float32))
            plt.imsave(os.path.join(d_gt, f"{base_name}.png"),   gt_m_full, cmap='jet')

            save_pfm(os.path.join(d_pred, f"{base_name}.pfm"),   pred_m_scaled.astype(np.float32))
            plt.imsave(os.path.join(d_pred, f"{base_name}.png"), pred_m_scaled, cmap='jet')

           
            err_percent = rel_percent_depth_difference_map(gt_m_full, pred_m_scaled)
            plt.figure()
            plt.imshow(err_percent, cmap='coolwarm', vmin=-100, vmax=100)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Percent Depth Error (%)')
            cbar.ax.text(0.25, 1.05, 'Predicted Closer', transform=cbar.ax.transAxes, ha='left', va='center')
            cbar.ax.text(0.25, -0.05, 'Predicted Farther', transform=cbar.ax.transAxes, ha='left', va='center')
            plt.axis('off')
            plt.savefig(os.path.join(d_err, f"percent_depth_error_map_{base_name}.png"),
                        bbox_inches='tight', pad_inches=0.05, dpi=200)
            plt.close()

    overall_abs_rel = float(np.mean(abs_rel_list)) if abs_rel_list else float("nan")
    overall_sq_rel  = float(np.mean(sq_rel_list))  if sq_rel_list  else float("nan")
    overall_rmse    = float(np.mean(rmse_list))    if rmse_list    else float("nan")
    overall_rmse_mm = overall_rmse * 1000.0
    overall_rmse_log= float(np.mean(rmse_log_list))if rmse_log_list else float("nan")
    overall_a1      = float(np.mean(a1_list))      if a1_list      else float("nan")

    print(f"Overall metrics -> RMSE (mm): {overall_rmse_mm:.6f}, "
          f"abs_rel: {overall_abs_rel:.6f}, sq_rel: {overall_sq_rel:.6f}, "
          f"a1: {overall_a1:.6f}")

    print("-> Saved outputs under:", outputs_root)
    print("-> Done.")

if __name__ == "__main__":
    opt = MonodepthOptions().parse()
    evaluate(opt)
