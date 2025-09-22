
import argparse
from pathlib import Path
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F



def load_image_cpu(path): 
    im = Image.open(path).convert('RGB')
    arr = np.array(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  
    return t

def load_mask_cpu(path, masked_if_nonzero=True):
    m = Image.open(path).convert('L')
    arr = (np.array(m, dtype=np.float32))/255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)    
    if masked_if_nonzero:
        t = (t > 0.5).float()   
    else:
        t = (t <= 0.5).float()
    return t

def save_image(tensor, path):
    t = tensor.clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()
    arr = (t*255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]



def make_base_coords(h, w, device):
    ys, xs = torch.meshgrid(torch.arange(h, device=device),
                            torch.arange(w, device=device), indexing='ij') 
    return torch.stack([xs, ys], dim=-1).float().unsqueeze(0)  

def pix2norm(coords, h, w): 
    x = coords[...,0] 
    y = coords[...,1] 
    nx = x / (w-1) * 2 - 1 
    ny = y / (h-1) * 2 - 1 
    return torch.stack([nx, ny], dim=-1) 

def sample_bilinear(img, coords_pix): 
    n,c,h,w = img.shape 
    grid = pix2norm(coords_pix, h, w) 
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True) 

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
  
    img1 = img1.to(device, non_blocking=True) 
    img2 = img2.to(device, non_blocking=True) 
    img1p, pad = pad_to_multiple(img1, 8, 'replicate') 
    img2p, _   = pad_to_multiple(img2, 8, 'replicate')  
    i1, i2 = tfm(img1p, img2p) 
    with torch.cuda.amp.autocast(enabled=amp): 
        flows = model(i1, i2) 
        flow  = flows[-1] 
    flow = unpad(flow, pad) 
    return flow  



@torch.inference_mode()
def fill_one_frame_streaming(
    idx, img_paths, mask_paths, model, tfm, device='cuda', 
    max_search=10, prefer='nearest', masked_if_nonzero=True, amp=True
):
    img_cur_cpu  = load_image_cpu(img_paths[idx]) 
    mask_cur_cpu = load_mask_cpu(mask_paths[idx], masked_if_nonzero)  
    H,W = img_cur_cpu.shape[-2:] 
    img_cur  = img_cur_cpu.to(device, non_blocking=True)   
    mask_cur = mask_cur_cpu.to(device, non_blocking=True) 

    masked = (mask_cur>0.5)                    
    if masked.sum()==0:
        return img_cur                          

    base_coords = make_base_coords(H, W, device) 

 
    forward_found = torch.zeros_like(masked, dtype=torch.bool) 
    backward_found= torch.zeros_like(masked, dtype=torch.bool)  
    forward_color = torch.zeros_like(img_cur) 
    backward_color= torch.zeros_like(img_cur) 
    forward_steps = torch.full_like(masked, 1e9, dtype=torch.float32) 
    backward_steps= torch.full_like(masked, 1e9, dtype=torch.float32) 


    cur_coords = base_coords.clone() 
    cur = idx 
    for step in range(1, max_search+1): 
        if cur >= len(img_paths)-1: break 
        
        img_a_cpu = load_image_cpu(img_paths[cur]) 
        img_b_cpu = load_image_cpu(img_paths[cur+1]) 
        flow = compute_flow_pair(model, tfm, img_a_cpu, img_b_cpu, device=device, amp=amp)  

        cur_coords = cur_coords + sample_flow(flow, cur_coords)  
        cur += 1 

        
        tgt_mask = load_mask_cpu(mask_paths[cur], masked_if_nonzero).to(device, non_blocking=True) 
        good = (sample_bilinear(tgt_mask, cur_coords) < 0.3) & (~forward_found) & masked 

        if good.any(): 
            img_cur_cpu2 = load_image_cpu(img_paths[cur]) 
            col = sample_bilinear(img_cur_cpu2.to(device, non_blocking=True), cur_coords)
            forward_color = torch.where(good.expand_as(forward_color), col, forward_color) 
            forward_found = torch.where(good, torch.ones_like(forward_found, dtype=torch.bool), forward_found) 
            forward_steps = torch.where(good, torch.full_like(forward_steps, float(step)), forward_steps) 

      
        del img_a_cpu, img_b_cpu, flow, tgt_mask 
        if 'cuda' in str(device): torch.cuda.empty_cache() 

        if (forward_found | (~masked)).all(): 
            break

    
    cur_coords = base_coords.clone() 
    cur = idx 
    for step in range(1, max_search+1): 
        if cur <= 0: break 
        img_a_cpu = load_image_cpu(img_paths[cur])     
        img_b_cpu = load_image_cpu(img_paths[cur-1])   
        flow = compute_flow_pair(model, tfm, img_a_cpu, img_b_cpu, device=device, amp=amp)  

        cur_coords = cur_coords + sample_flow(flow, cur_coords) 
        cur -= 1 

        tgt_mask = load_mask_cpu(mask_paths[cur], masked_if_nonzero).to(device, non_blocking=True) 
        good = (sample_bilinear(tgt_mask, cur_coords) < 0.3) & (~backward_found) & masked 

        if good.any(): 
            img_cur_cpu2 = load_image_cpu(img_paths[cur]) 
            col = sample_bilinear(img_cur_cpu2.to(device, non_blocking=True), cur_coords) 
            backward_color = torch.where(good.expand_as(backward_color), col, backward_color) 
            backward_found = torch.where(good, torch.ones_like(backward_found, dtype=torch.bool), backward_found) 
            backward_steps = torch.where(good, torch.full_like(backward_steps, float(step)), backward_steps) 

        del img_a_cpu, img_b_cpu, flow, tgt_mask 
        if 'cuda' in str(device): torch.cuda.empty_cache() 

        if (backward_found | (~masked)).all(): 
            break 

    
    out = img_cur.clone() 
    use_fwd_only = forward_found & (~backward_found)  
    use_bwd_only = backward_found & (~forward_found)    
    use_both     = forward_found & backward_found 

    if prefer == 'nearest': 
        if use_both.any(): 
            fw_nearer = forward_steps < backward_steps 
            out = torch.where((use_both & fw_nearer).expand_as(out), forward_color, out) 
            out = torch.where((use_both & (~fw_nearer)).expand_as(out), backward_color, out) 
    else:  
        if prefer == 'forward': 
            out = torch.where(use_both.expand_as(out), forward_color, out) 
        else:
            out = torch.where(use_both.expand_as(out), backward_color, out) 

    out = torch.where(use_fwd_only.expand_as(out), forward_color, out) 
    out = torch.where(use_bwd_only.expand_as(out), backward_color, out) 
    out = torch.where(masked.expand_as(out), out, img_cur)  
    return out  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, default='musk/final_musk/cecum_t1_a')
    parser.add_argument('--image_glob', type=str, default='*0.png')
    parser.add_argument('--mask_glob',  type=str, default='*_mask.001.png')
    parser.add_argument('--out_dir',    type=str, default='musk/inpaint_specular/cecum_t1_a')
    parser.add_argument('--device',     type=str, default='cuda')
    parser.add_argument('--max_search', type=int, default=100)
    parser.add_argument('--raft_large', action='store_true')
    parser.add_argument('--masked_if_nonzero', action='store_true', default=True,
                        help='--no-masked_if_nonzero')
    parser.add_argument('--no_amp', action='store_true', help='close AMP, save GPU memory when using large models')
    args = parser.parse_args()
    
    frames_dir = Path(args.frames_dir)
    img_paths  = sorted(frames_dir.glob(args.image_glob), key=natural_key)
    mask_paths = sorted(frames_dir.glob(args.mask_glob),  key=natural_key)
    assert len(img_paths)==len(mask_paths) and len(img_paths)>=2, "need >=2 frames & masks"

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

   
    H,W = np.array(Image.open(img_paths[0]).convert('RGB')).shape[:2]

    model, tfm = load_raft(device=device, large=args.raft_large)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

   
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