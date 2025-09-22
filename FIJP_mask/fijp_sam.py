
import argparse
import os
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from pathlib import Path

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    raise ImportError("segment‑anything not found. Install with `pip install git+https://github.com/facebookresearch/segment‑anything.git`. ")



def preprocess_bgr(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
    l, a, b = cv2.split(lab) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l) 
    lab = cv2.merge((l, a, b)) 
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    
    inv_gamma = 1.0 / gamma 
    table = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype("uint8") 
    img = cv2.LUT(img, table) 
    return img


def iou_and_inter(box1, box2):
   
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-6
    return inter / union, inter


def suppress_nested_boxes(boxes,
                          iou_thresh: float = 0.5,
                          contain_thresh: float = 0.9):
   
    if not boxes:
        return [], []

    boxes_arr = np.asarray(boxes, dtype=np.float32)
    areas = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])
    order = areas.argsort()[::-1]             

    keep_idx = []
    for idx in order:
        box_cur = boxes_arr[idx]
        discard = False
        for kept in keep_idx:
            box_big = boxes_arr[kept]
            iou, inter = iou_and_inter(box_cur, box_big)

           
            if iou >= iou_thresh:
                discard = True
                break

            
            area_cur = areas[idx]
            overlap_ratio_small = inter / area_cur
            if overlap_ratio_small >= contain_thresh:
                discard = True
                break

        if not discard:
            keep_idx.append(idx)

    new_boxes = [boxes[i] for i in keep_idx]
    return new_boxes, keep_idx

def highpass_fft(v: np.ndarray, cutoff_ratio: float = 0.2, order: int = 2) -> np.ndarray:
  
    h, w = v.shape

    F = np.fft.fftshift(np.fft.fft2(v)) 
    u, vgrid = np.meshgrid(np.arange(-w // 2, w // 2), np.arange(-h // 2, h // 2)) 
    D = np.sqrt(u ** 2 + vgrid ** 2) 
    D0 = cutoff_ratio * min(h, w) / 2.0  
    with np.errstate(divide="ignore", invalid="ignore"): 
        H_hp = 1.0 / (1.0 + (D0 / (D + 1e-6)) ** (2 * order)) 
    H_hp = np.fft.ifftshift(H_hp) 
    F_hp = F * H_hp 
    resp = np.abs(np.fft.ifft2(F_hp)) 
    return resp

def bright_mask_percentile(V, p=97):
    thresh = np.percentile(V, p) 
    return (V >= thresh).astype(np.uint8) * 255

def fijp_heatmap(img_bgr: np.ndarray,
                 cutoff_ratio: float = 0.5, 
                 k_bright: float = 1.0) -> np.ndarray: 
   
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) 
    V, S = hsv[:,:,2].astype(np.float32), hsv[:,:,1].astype(np.float32) 

    
    F_hp = highpass_fft(V, cutoff_ratio=cutoff_ratio) 
   
    F_norm = cv2.normalize(F_hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) 

    bright_mask = bright_mask_percentile(V, p=95)  
   
   
    return F_norm, bright_mask

def low_saturation_mask(img_bgr: np.ndarray, s_thresh: int = 40) -> np.ndarray:
   
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]                     
    return (S < s_thresh).astype(np.uint8) * 255

def generate_prompts(img_bgr: np.ndarray,
                     img_path: str,
                     cutoff_ratio: float = 0.2,
                     k_bright: float = 1.0,
                     morph_kernel: int = 10,
                     min_area: int = 5,
                     save_heat_path: str = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
  
    if not os.path.exists(save_heat_path):
        os.makedirs(save_heat_path)

    F_norm, bright_mask = fijp_heatmap(img_bgr, cutoff_ratio, k_bright) 

    F_norm_p = os.path.join(save_heat_path, os.path.basename(img_path).split('.')[0] + '_F_norm.png') 
    _, F_norm = cv2.threshold(F_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

    bright_mask_p = os.path.join(save_heat_path, os.path.basename(img_path).split('.')[0] + '_bright_mask.png') 
    _, bright_mask = cv2.threshold(bright_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

    color_mask = low_saturation_mask(img_bgr, s_thresh=40) 
    color_mask_p = os.path.join(save_heat_path, os.path.basename(img_path).split('.')[0] + '_color_mask.png') 

    heat = cv2.bitwise_and(F_norm, cv2.bitwise_and(bright_mask, color_mask)) 
    
    _, bina_bin = cv2.threshold(heat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bina_p = os.path.join(save_heat_path, os.path.basename(img_path).split('.')[0] + '_bina.png')
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)) 
    heat_bin = cv2.morphologyEx(bina_bin, cv2.MORPH_DILATE, kernel) 
    heat_bin_p = os.path.join(save_heat_path, os.path.basename(img_path).split('.')[0] + '_heat_bin.png')
  

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(heat_bin) 

    points, boxes = [], [] 
    for i in range(1, num):  
        x, y, w, h, area = stats[i] 
        if area < min_area: 
            continue
        cx, cy = centroids[i] 
        points.append([cx, cy]) 
        boxes.append([x, y, x + w, y + h]) 

    nms_boxes, keep_idx = suppress_nested_boxes(boxes, iou_thresh=0.5) 
    nms_points = [points[i] for i in keep_idx] 
    for box in nms_boxes:
        x1, y1, x2, y2 = box
        nms_box = cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) 
    nms_p = os.path.join(save_heat_path, os.path.basename(img_path).split('.')[0] + '_nms.png')
   
   


    if points:
        return np.array(nms_points, dtype=np.float32), np.array(nms_boxes, dtype=np.float32) 
    else:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 4), dtype=np.float32) 
    

def auto_params(img_bgr, k0=1.0, alpha=0.5, beta=0.8):
 
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) 
    V = hsv[:,:,2].astype(np.float32) 


    mu, sigma = V.mean(), V.std() 
    sigma_ref = 40  
    k_bright = k0 + alpha * (sigma - sigma_ref) / sigma_ref 
    k_bright = float(np.clip(k_bright, 0.4, 2.0)) 


    F = np.fft.fftshift(np.fft.fft2(V)) 
    A = np.abs(F) 
    h, w = V.shape 

    y, x = np.indices((h, w)) 
    r = np.sqrt((x - w/2)**2 + (y - h/2)**2).astype(np.int32) 
    r = np.clip(r, 0, int(max(h, w)))  
    radial_energy = np.bincount(r.ravel(), A.ravel())   
    cumsum = np.cumsum(radial_energy) 
    R90 = np.searchsorted(cumsum, 0.90 * cumsum[-1])
    cutoff_ratio = beta * R90 / (0.5 * min(h, w)) 
    cutoff_ratio = float(np.clip(cutoff_ratio, 0.12, 0.35)) 

    return cutoff_ratio, k_bright


def load_sam(checkpoint: str, model_type: str = "vit_b", device: str = "cuda") -> SamPredictor:

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def segment_with_sam(img_bgr: np.ndarray,
                     predictor: SamPredictor,
                     points: np.ndarray,
                     boxes: np.ndarray,
                     multimask_output: bool = False) -> np.ndarray:

    predictor.set_image(img_bgr) 
    H, W = img_bgr.shape[:2]
    final_mask = np.zeros((H, W), dtype=np.uint8)

    
    if boxes.size: 
  
        for i, box in enumerate(boxes):
         
            pt = points[i:i+1] if points.shape[0] > i else None
            lbl = np.array([1], dtype=np.int32) if pt is not None else None
            
            mask_i, _, _ = predictor.predict(
                box=box[None, :],               
                point_coords=pt,
                point_labels=lbl,
                multimask_output=multimask_output,
            )
            final_mask |= mask_i[0].astype(np.uint8) 
    final_mask = final_mask * 255

    return final_mask, img_bgr



def save_mask(mask: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)


def glob_imgs(path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if path.is_dir():

        return sorted(path.rglob('*.bmp'))
    elif path.is_file() and path.suffix.lower() in exts:
        return [path]
    else:
        raise FileNotFoundError(f"No images found in {path}")



def process_images(paths: List[Path], predictor: SamPredictor, out_dir: Path, args):
    for img_path in tqdm(paths, desc="Processing images"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to read {img_path}")
            continue

        img = preprocess_bgr(img, gamma=args.gamma)
        img_raw  = cv2.imread(str(img_path))

        if args.auto_params:
            cutoff_ratio, k_bright = auto_params(img)
        else:
            cutoff_ratio, k_bright = args.cutoff_ratio, args.k_bright
   

        points, boxes = generate_prompts(
            img,
            img_path,
            cutoff_ratio=cutoff_ratio,
            k_bright=k_bright,
            morph_kernel=args.morph_kernel,
            min_area=args.min_area,
            save_heat_path=args.save_heat_path
        )

        mask,img = segment_with_sam(img_raw, predictor, points, boxes) 

        rel = img_path.stem + "_mask.001.png" 
        rel_img = img_path.stem + ".png" 
        save_mask(mask, out_dir / rel) 
        save_mask(img, out_dir / rel_img) 


def process_video(path: Path, predictor: SamPredictor, out_dir: Path, args):
    cap = cv2.VideoCapture(str(path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_p = out_dir / (path.stem + "_mask.mp4")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height), isColor=False)

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None, desc="Processing video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_prep = preprocess_bgr(frame, gamma=args.gamma)
        points, boxes = generate_prompts(
            frame_prep,
            cutoff_ratio=args.cutoff_ratio,
            k_bright=args.k_bright,
            morph_kernel=args.morph_kernel,
            min_area=args.min_area,
        )
        mask = segment_with_sam(frame_prep, predictor, points, boxes)
        writer.write(mask)
        pbar.update(1)
    pbar.close()
    cap.release()
    writer.release()



def parse_args():
    parser = argparse.ArgumentParser(description="FIJP + SAM specular‑highlight segmentation")
    in_group = parser.add_mutually_exclusive_group(required=True)
    in_group.add_argument("--images", type=str, help="Path to image or directory of images")
    in_group.add_argument("--video", type=str, help="Path to a video file")

    parser.add_argument("--sam_ckpt", type=str, default="./pretrain/sam_vit_b_01ec64.pth", help="Path to SAM checkpoint (.pth)")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="SAM backbone")
    parser.add_argument("--device", type=str, default="cuda", help="torch device (cuda or cpu)")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to save masks")
    parser.add_argument("--save_heat_path", type=str, default="heatmaps", help="Directory to save heatmaps")


    parser.add_argument("--cutoff_ratio", type=float, default=0.1, help="High‑pass cutoff ratio")
    parser.add_argument("--k_bright", type=float, default=2.0, help="Brightness threshold multiplier (mean + k*std)")
    parser.add_argument("--auto_params", action="store_true",default=True, help="Use adaptive parameters for cutoff_ratio and k_bright")
    parser.add_argument("--morph_kernel", type=int, default=20, help="Morphological kernel diameter")
    parser.add_argument("--min_area", type=int, default=50, help="Min connected‑component area in px")

   
    parser.add_argument("--gamma", type=float, default=1.2, help="Gamma for gamma correction")

    return parser.parse_args()



def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        
    predictor = load_sam(args.sam_ckpt, args.model_type, args.device)

    if args.images:
        img_paths = glob_imgs(Path(args.images))
        process_images(img_paths, predictor, out_dir, args)
    else:  
        process_video(Path(args.video), predictor, out_dir, args)

    print("\u2713 Done.")


if __name__ == "__main__":
    main()
