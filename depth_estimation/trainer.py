from __future__ import absolute_import, division, print_function

import os
import time
import json
import datasets
import networks
from networks import PPSNet_Backbone, PPSNet_Refinement
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import optical_flow_funs as OF
class Trainer:
    def __init__(self, options):
        self.opt = options
        print(self.opt.log_dir)
        print(self.opt.model_name)
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

     
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.parameters_to_train_1 = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        if self.opt.dataset.lower() == "c3vd" and not self.opt.pps_pretrained_ckpt:
            raise ValueError("--pps_pretrained_ckpt is required for dataset=c3vd")
       
        if self.opt.dataset.lower() == "c3vd":
            self.models["pps_backbone"] = PPSNet_Backbone.from_pretrained(
                'LiheYoung/depth_anything_vits14'
            ).to(self.device)

         
            token_dim = self.models["pps_backbone"].pretrained.blocks[0].attn.qkv.in_features
            self.models["pps_refine"] = PPSNet_Refinement(
                disparity_channels=1, encoded_dim=384, feature_dim=token_dim, heads=8
            ).to(self.device)
            self.parameters_to_train += list(self.models["pps_refine"].parameters())

            
            self.models["decompose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["decompose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["decompose_encoder"].parameters())

            self.models["decompose"] = networks.decompose_decoder(
                self.models["decompose_encoder"].num_ch_enc, self.opt.scales)
            self.models["decompose"].to(self.device)
            self.parameters_to_train += list(self.models["decompose"].parameters())

            self.models["adjust_net"] = networks.adjust_net()
            self.models["adjust_net"].to(self.device)
            self.parameters_to_train += list(self.models["adjust_net"].parameters())

            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)
            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())
            if not hasattr(self.opt, "gt_depth_weight"):
                self.opt.gt_depth_weight = 1.0
            if not hasattr(self.opt, "depth_consistency"):
                self.opt.depth_consistency = 1.0

        else:
        
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())

            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())

            self.models["decompose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["decompose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["decompose_encoder"].parameters())

            self.models["decompose"] = networks.decompose_decoder(
                self.models["decompose_encoder"].num_ch_enc, self.opt.scales)
            self.models["decompose"].to(self.device)
            self.parameters_to_train += list(self.models["decompose"].parameters())

            self.models["adjust_net"] = networks.adjust_net()
            self.models["adjust_net"].to(self.device)
            self.parameters_to_train += list(self.models["adjust_net"].parameters())

            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)
            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())


       
        ckpt_path = os.path.expanduser(self.opt.pps_pretrained_ckpt)
        assert os.path.isfile(ckpt_path), f"PPS ckpt not found: {ckpt_path}"
        print(f"Loading PPSNet pretrained weights from: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

   
        if "student_state_dict" in ckpt:
            backbone_state_dict = {}
            for k, v in ckpt["student_state_dict"].items():
                name = k[7:] if k.startswith("module.") else k
                backbone_state_dict[name] = v
            self.models["pps_backbone"].load_state_dict(backbone_state_dict, strict=True)
        else:
           
            self.models["pps_backbone"].load_state_dict(ckpt, strict=True)

      
        if "refiner_state_dict" in ckpt:
            ref_state = {}
            for k, v in ckpt["refiner_state_dict"].items():
                name = k[7:] if k.startswith("module.") else k
                ref_state[name] = v
            self.models["pps_refine"].load_state_dict(ref_state, strict=True)
        else:
            print("[warn] No refiner_state_dict in ckpt; PPSNet_Refinement will stay randomly initialized.")

        if getattr(self.opt, "freeze_pps_backbone", False): 
            self.models["pps_backbone"].eval()
            for p in self.models["pps_backbone"].parameters():
                p.requires_grad = False
        else:
       
            self.parameters_to_train += list(self.models["pps_backbone"].parameters())
        def _freeze_bn_stats(model):
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()                 
                    m.train = lambda mode=True: None  
        _freeze_bn_stats(self.models["pps_refine"])

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
        self.model_optimizer, [self.opt.scheduler_step_size], 0.1)
        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)


   
        datasets_dict = {
            "endovis": datasets.SCAREDRAWDataset,
            "c3vd": datasets.C3VD_Dataset,   
            "C3VD": datasets.C3VD_Dataset,   
        }
        assert self.opt.dataset in datasets_dict, f"Unknown dataset {self.opt.dataset}"
        self.dataset = datasets_dict[self.opt.dataset]


        if self.opt.dataset.lower() == "c3vd":
      
            default_root = os.path.join(os.path.dirname(__file__), "datasets", "C3VD_splits")
            train_list = getattr(self.opt, "c3vd_train_list",
                                 os.path.join(default_root, "train.txt"))
            val_list   = getattr(self.opt, "c3vd_val_list",
                                 os.path.join(default_root, "val.txt"))

            
            train_dataset = self.dataset(self.opt.data_path, train_list, mode="Train")
            val_dataset   = self.dataset(self.opt.data_path, val_list,   mode="Val")

         
            if self.num_input_frames > 1:
                print("self.num_input_frames > 1")
        else:
    
            fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
            train_filenames = readlines(fpath.format("train"))
            val_filenames = readlines(fpath.format("val"))
            img_ext = '.png'

            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
            val_dataset = self.dataset(
                self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)

        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        print("batch_size: {}".format(self.opt.batch_size))
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w).to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w).to(self.device)

        print("Using dataset:\n  ", self.opt.dataset)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()


    def _adapt_c3vd_batch(self, inputs_raw):
       
        out = {}

        
        img = inputs_raw["image"]  
        
        out[("color", 0, 0)] = img
        out[("color_aug", 0, 0)] = img  
       
      
        K3 = inputs_raw["intrinsics"]
        if not torch.is_tensor(K3):
            K3 = torch.as_tensor(K3)
        if K3.dim() == 2:
            K3 = K3.unsqueeze(0) 
        B = K3.shape[0]
        K = torch.zeros(B, 4, 4, dtype=K3.dtype, device=K3.device)
        K[:, :3, :3] = K3
        K[:, 3, 3] = 1.0
        inv_K = torch.inverse(K)

        out[("K", 0)] = K
        out[("inv_K", 0)] = inv_K

      
        for k in ["depth", "normal", "mask", "n_intrinsics",
                  "translation_vector", "light_data", "dataset", "id"]:
            if k in inputs_raw:
                out[k] = inputs_raw[k]

        return out

    def _predict_pps_disp_depth(self, img, n_intrinsics, light_data):
        B, _, H, W = img.shape
        ref_dirs = OF.get_camera_pixel_directions((H, W), n_intrinsics, normalized_intrinsics=True).to(self.device)
        light_pos, light_dir, mu = light_data
        disp_coarse, rgb_feats, colored_feats = self.models["pps_backbone"](
            img, ref_dirs, light_pos, light_dir, mu, n_intrinsics.to(self.device)
        )
        disp_refined = self.models["pps_refine"](rgb_feats, colored_feats, disp_coarse)  
        disp = disp_refined.unsqueeze(1) 

   
        depth = 1.0 / (disp + 1e-6)
        depth = torch.clamp(depth, 0, 1)
       
        dmax = depth.flatten(2).max(-1).values.view(B, 1, 1, 1).clamp_min(1e-6)
        depth = depth / dmax

        return disp, depth

    def set_train(self):
        train_names = ["decompose_encoder","decompose","adjust_net","pose_encoder","pose","pps_refine"]
        if self.opt.dataset.lower() != "c3vd":
            train_names += ["encoder","depth"]
        elif not getattr(self.opt, "freeze_pps_backbone", False):
            train_names += ["pps_backbone"]

        
        for n in train_names:
            self.models[n].train()
            for p in self.models[n].parameters():
                p.requires_grad = True

 
        if self.opt.dataset.lower() == "c3vd" and getattr(self.opt, "freeze_pps_backbone", False):
            self.models["pps_backbone"].eval()
            for p in self.models["pps_backbone"].parameters():
                p.requires_grad = False

    def set_eval(self):
        names = ["decompose_encoder","decompose","adjust_net","pose_encoder","pose"]
        if self.opt.dataset.lower() == "c3vd":
            names += ["pps_backbone","pps_refine"]
        else:
            names += ["encoder","depth"]
        for n in names:
            self.models[n].eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training")
        print(self.model_optimizer.param_groups[0]['lr'])
        for batch_idx, inputs in enumerate(self.train_loader):
           
       
            before_op_time = time.time()

        
            if self.opt.dataset.lower() == "c3vd":
                inputs = self._adapt_c3vd_batch(inputs)

            self.set_train()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            phase = batch_idx % self.opt.log_frequency == 0

            if phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        for key, ipt in list(inputs.items()):
            if torch.is_tensor(ipt):
                inputs[key] = ipt.to(self.device)

        outputs = {}

        if self.opt.dataset.lower() == "c3vd":
            outputs = {}

         
            img0 = inputs[("color_aug", 0, 0)]
            B, _, H, W = img0.shape
            light_pos, light_dir, mu = inputs["light_data"]

            def to_tensor(x):
                return x if torch.is_tensor(x) else torch.as_tensor(x)
            light_pos = to_tensor(light_pos)
            light_dir = to_tensor(light_dir)
            mu        = to_tensor(mu)

            if isinstance(light_pos, (list, tuple)):
                light_pos = torch.stack([to_tensor(t) for t in light_pos], dim=0)
            if isinstance(light_dir, (list, tuple)):
                light_dir = torch.stack([to_tensor(t) for t in light_dir], dim=0)
            if isinstance(mu, (list, tuple)):
                mu = torch.stack([to_tensor(t) for t in mu], dim=0)

            light_pos = light_pos.to(device=self.device, dtype=img0.dtype)
            light_dir = light_dir.to(device=self.device, dtype=img0.dtype)
            mu        = mu.to(device=self.device, dtype=img0.dtype)

            if light_pos.dim() == 1:
                light_pos = light_pos.unsqueeze(0).expand(B, -1)
            if light_dir.dim() == 1:
                light_dir = light_dir.unsqueeze(0).expand(B, -1)
            if mu.dim() == 0:
                mu = mu.expand(B).clone()
            elif mu.dim() == 1 and mu.shape[0] == 1:
                mu = mu.expand(B).clone()

            light_data = (light_pos, light_dir, mu)

            for f_i in self.opt.frame_ids:
                img_f = inputs[("color_aug", f_i, 0)]  
                disp_f, depth_f = self._predict_pps_disp_depth(img_f, inputs["n_intrinsics"], light_data)

          
                if (disp_f.shape[-2:] != (self.opt.height, self.opt.width)):
                    disp_f  = F.interpolate(disp_f, size=(self.opt.height, self.opt.width), mode="bilinear", align_corners=True)
                    depth_f = F.interpolate(depth_f, size=(self.opt.height, self.opt.width), mode="bilinear", align_corners=True)

               
                if f_i == 0:
                    outputs[("disp", 0)] = disp_f 
                outputs[("disp", f_i)]  = disp_f
                outputs[("depth", f_i)] = depth_f

        else:
            
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)
            _, depth0 = disp_to_depth(outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0)] = depth0

     
        outputs.update(self.predict_poses(inputs))
        self.decompose(inputs, outputs)

        losses = self.compute_losses(inputs, outputs)
        return outputs, losses


    def predict_poses(self, inputs):
        outputs = {}
        if self.num_pose_frames == 2 and len(self.opt.frame_ids) > 1:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids} 
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        inputs_all = [pose_feats[f_i], pose_feats[0]]
                    else:
                        inputs_all = [pose_feats[0], pose_feats[f_i]]
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        return outputs

    def decompose(self, inputs, outputs):
        for f_i in self.opt.frame_ids:
            decompose_features = self.models["decompose_encoder"](inputs[("color_aug", f_i, 0)])
            refl, light = self.models["decompose"](decompose_features)
            outputs[("reflectance", 0, f_i)] = refl
            outputs[("light", 0, f_i)] = light
            outputs[("reprojection_color", 0, f_i)] = refl * light


        if len(self.opt.frame_ids) > 1:
            disp = outputs[("disp", 0)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            for frame_id in self.opt.frame_ids[1:]:
                T = outputs[("cam_T_cam", 0, frame_id)]
                cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
                pix_coords = self.project_3d[0](cam_points, inputs[("K", 0)], T)
                outputs[("warp", 0, frame_id)] = pix_coords

                outputs[("reflectance_warp", 0, frame_id)] = F.grid_sample(
                    outputs[("reflectance", 0, frame_id)], pix_coords,
                    padding_mode="border", align_corners=True)

                outputs[("light_warp", 0, frame_id)] = F.grid_sample(
                    outputs[("light", 0, frame_id)], pix_coords,
                    padding_mode="border", align_corners=True)

                outputs[("color_warp", 0, frame_id)] = F.grid_sample(
                    inputs[("color_aug", frame_id, 0)], pix_coords,
                    padding_mode="border", align_corners=True)

                mask_ones = torch.ones_like(inputs[("color_aug", frame_id, 0)])
                mask_warp = F.grid_sample(mask_ones, pix_coords, padding_mode="zeros", align_corners=True)
                valid_mask = (mask_warp.abs().mean(dim=1, keepdim=True) > 0.0).float()
                outputs[("valid_mask", 0, frame_id)] = valid_mask

                outputs[("warp_diff_color", 0, frame_id)] = torch.abs(
                    inputs[("color_aug", 0, 0)] - outputs[("color_warp", 0, frame_id)]
                ) * valid_mask

                outputs[("transform", 0, frame_id)] = self.models["adjust_net"](outputs[("warp_diff_color", 0, frame_id)])
                outputs[("light_adjust_warp", 0, frame_id)] = torch.clamp(
                    outputs[("transform", 0, frame_id)] + outputs[("light_warp", 0, frame_id)],
                    min=0.0, max=1.0
                )
                outputs[("reprojection_color_warp", 0, frame_id)] = \
                    outputs[("reflectance_warp", 0, frame_id)] * outputs[("light_adjust_warp", 0, frame_id)]

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss
    def _gt_supervision_loss(self, inputs, outputs):
        
        if "depth" not in inputs:
            return torch.tensor(0.0, device=self.device, dtype=outputs[("disp", 0)].dtype)

        pred_depth = outputs.get(("depth", 0), None)
        if pred_depth is None:
           
            disp0 = outputs[("disp", 0)]
            _, pred_depth = disp_to_depth(disp0, self.opt.min_depth, self.opt.max_depth)

        gt = inputs["depth"]             
        mask = inputs.get("mask", None)  

        
        if gt.shape[-2:] != (self.opt.height, self.opt.width):
            gt = F.interpolate(gt, size=(self.opt.height, self.opt.width), mode="nearest")
        if mask is not None and mask.shape[-2:] != (self.opt.height, self.opt.width):
            mask = F.interpolate(mask, size=(self.opt.height, self.opt.width), mode="nearest")

        
        valid = (gt > self.opt.min_depth) & (gt < self.opt.max_depth)
        if mask is not None:
            valid = valid & (mask > 0.5)

       
        l1 = torch.abs(pred_depth - gt)
        if valid.any():
            loss = l1[valid].mean()
        else:
            loss = torch.tensor(0.0, device=self.device, dtype=pred_depth.dtype)

        return loss
    def _depth_consistency_loss(self, inputs, outputs):
       
        if len(self.opt.frame_ids) <= 1:
            return torch.tensor(0.0, device=self.device, dtype=outputs[("disp", 0)].dtype)

       
        depth_t = outputs[("depth", 0)]  
        inv_K_t = inputs[("inv_K", 0)]
        K_t     = inputs[("K", 0)]

        B, _, H, W = depth_t.shape
        loss = 0.0
        valid_pair_count = 0

       
        cam_points_t = self.backproject_depth[0](depth_t, inv_K_t)  

        for s in self.opt.frame_ids[1:]:
          
            T_t2s = outputs.get(("cam_T_cam", 0, s), None) 
            if T_t2s is None:
                continue

      
            pix_t2s = self.project_3d[0](cam_points_t, K_t, T_t2s)  
            
            mask_ones = torch.ones((B, 1, H, W), device=self.device, dtype=depth_t.dtype)
            valid_mask = F.grid_sample(mask_ones, pix_t2s, padding_mode="zeros", align_corners=True)
            valid_mask = (valid_mask > 0.0).float() 

            
            depth_s = outputs.get(("depth", s), None)
            if depth_s is None:
                
                continue
            depth_s_at_t = F.grid_sample(depth_s, pix_t2s, padding_mode="border", align_corners=True) 

           
            P_s = torch.matmul(T_t2s, cam_points_t)  
            Z_s_pred = P_s[:, 2:3, :]               
            Z_s_pred = Z_s_pred.view(B, 1, H, W)    

      
            diff = torch.abs(depth_s_at_t - Z_s_pred) * valid_mask
            denom = valid_mask.sum() + 1e-7
            loss += diff.sum() / denom
            valid_pair_count += 1

        if valid_pair_count > 0:
            loss = loss / valid_pair_count
        else:
            loss = torch.tensor(0.0, device=self.device, dtype=depth_t.dtype)

        return loss


    def compute_losses(self, inputs, outputs):
        losses = {}

        disp0 = outputs[("disp", 0)]
        device = disp0.device
        dtype  = disp0.dtype
        z = torch.tensor(0.0, device=device, dtype=dtype)
        
        total_loss          = z.clone()
        loss_reflec         = z.clone()
        loss_reprojection   = z.clone()
        loss_disp_smooth    = z.clone()
        loss_reconstruction = z.clone()

        
        for frame_id in self.opt.frame_ids:
            loss_reconstruction += self.compute_reprojection_loss(
                inputs[("color_aug", frame_id, 0)],
                outputs[("reprojection_color", 0, frame_id)]
            ).mean()

       
        if len(self.opt.frame_ids) > 1:
            for frame_id in self.opt.frame_ids[1:]:
                mask = outputs[("valid_mask", 0, frame_id)]
                loss_reflec += (torch.abs(
                    outputs[("reflectance", 0, 0)] - outputs[("reflectance_warp", 0, frame_id)]
                ).mean(1, True) * mask).sum() / (mask.sum() + 1e-7)

                loss_reprojection += (self.compute_reprojection_loss(
                    inputs[("color_aug", 0, 0)], outputs[("reprojection_color_warp", 0, frame_id)]
                ) * mask).sum() / (mask.sum() + 1e-7)

        disp = outputs[("disp", 0)]
        color = inputs[("color_aug", 0, 0)]
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        loss_disp_smooth = get_smooth_loss(norm_disp, color)
        loss_depth_cons = self._depth_consistency_loss(inputs, outputs)
        loss_gt = self._gt_supervision_loss(inputs, outputs)
 
        total_loss = (self.opt.reprojection_constraint * (loss_reprojection / max(1, len(self.opt.frame_ids) - 1))
                      + self.opt.reflec_constraint * (loss_reflec / max(1, len(self.opt.frame_ids) - 1))
                      + self.opt.disparity_smoothness * loss_disp_smooth
                      + self.opt.reconstruction_constraint * (loss_reconstruction / len(self.opt.frame_ids))
                      + self.opt.depth_consistency       * loss_depth_cons 
                      + self.opt.gt_depth_weight         * loss_gt  
                      )

        losses["loss"] = total_loss
        losses["loss_depth_cons"] = loss_depth_cons.detach()
        losses["loss_gt"] = loss_gt.detach()
        losses["loss_reprojection"]   = loss_reprojection.detach()
        losses["loss_reflec"]         = loss_reflec.detach()
        losses["loss_disp_smooth"]    = loss_disp_smooth.detach()
        losses["loss_reconstruction"] = loss_reconstruction.detach()
        return losses

    def val(self):
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            if self.opt.dataset.lower() == "c3vd":
                inputs = self._adapt_c3vd_batch(inputs)
            outputs, losses = self.process_batch(inputs)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f} | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
      

        for j in range(min(4, self.opt.batch_size)):
            writer.add_image("disp/{}".format(j), visualize_depth(outputs[("disp", 0)][j]), self.step)
            writer.add_image("input/{}".format(j), inputs[("color", 0, 0)][j].data, self.step)
      
            

    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        torch.save(self.model_optimizer.state_dict(),
                   os.path.join(save_folder, "adam.pth"))

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location="cpu")
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
