# Our proposed PPSNet, which includes a depth estimation backbone and a depth refinement module
# Part of this code is inspired by the Depth Anything (CVPR 2024) paper's code implementation
# See https://github.com/LiheYoung/Depth-Anything/blob/main/depth_anything/dpt.py

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from .blocks import FeatureFusionBlock, _make_scratch
# from losses.calculate_PPL import calculate_per_pixel_lighting
from .unet import UNet

import numpy as np
import matplotlib.colors as mcolors
import optical_flow_funs as OF

def calculate_per_pixel_lighting(pc_preds, light_pos, light_dir, mu):
    #pc_gt and pc_preds (b,m,n,3)
    #light_pos (b,3)
    #light_dir (b,3)
    #angular attenuation mu (b,)
    #return (b,m,n,3) (b,m,n,1)

    # Calculate PPL for pc_preds 从pc_preds计算PPL
    to_light_vec_preds = light_pos.unsqueeze(1).unsqueeze(1) - pc_preds  # 光源到点云的向量（x-p）
    n_to_light_vec_preds = F.normalize(to_light_vec_preds,dim=3) # 归一化(L(x))
    #(b,m,n,1)
    len_to_light_vec_preds = torch.norm(to_light_vec_preds,dim=3,keepdim=True) # 光源到点云的长度（|x-p|）
    light_dir_dot_to_light_preds = torch.sum(-n_to_light_vec_preds*light_dir.unsqueeze(1).unsqueeze(1),dim=3,keepdim=True).clamp(min=1e-8) # 光源方向和点云到光源向量的点积（-L(x)·l）
    numer_preds = torch.pow(light_dir_dot_to_light_preds, mu.view(-1,1,1,1)) # (L(x)·l)^mu
    atten_preds = numer_preds/(len_to_light_vec_preds**2).clamp(min=1e-8) # (|x-p|^2)

    return n_to_light_vec_preds, atten_preds

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

def _rgb_to_grayscale(input_tensor):
    # Assuming input_tensor is of shape (B, C, H, W) and C=3 for RGB
    return torch.matmul(input_tensor.permute(0, 2, 3, 1), torch.tensor([0.2989, 0.5870, 0.1140], device=input_tensor.device)).unsqueeze(1)

def _normalize_vectors(v):
    """ Normalize a batch of 3D vectors to unit vectors using PyTorch. """
    norms = v.norm(p=2, dim=1, keepdim=True)
    return v / (norms + 1e-8)  # Adding a small epsilon to avoid division by zero

def _image_derivatives(image,diff_type='center'):
    c = image.size(1)
    if diff_type=='center':
        sobel_x = 0.5*torch.tensor([[0.0,0,0],[-1,0,1],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
        sobel_y = 0.5*torch.tensor([[0.0,1,0],[0,0,0],[0,-1,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
    elif diff_type=='forward':
        sobel_x = torch.tensor([[0.0,0,0],[0,-1,1],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
        sobel_y = torch.tensor([[0.0,1,0],[0,-1,0],[0,0,0]],device=image.device).view(1,1,3,3).repeat(c,1,1,1)
    
    dp_du = torch.nn.functional.conv2d(image,sobel_x,padding=1,groups=3)
    dp_dv = torch.nn.functional.conv2d(image,sobel_y,padding=1,groups=3)
    return dp_du, dp_dv

def _point_cloud_to_normals(pc, diff_type='center'):
    #pc (b,3,m,n)
    #return (b,3,m,n)
    dp_du, dp_dv = _image_derivatives(pc,diff_type=diff_type)
    normal = torch.nn.functional.normalize( torch.cross(dp_du,dp_dv,dim=1)) # 计算法向量（
    return normal

def _get_normals_from_depth(depth, intrinsics, depth_is_along_ray=False , diff_type='center', normalized_intrinsics=True):
    #depth (b,1,m,n) 深度图
    #intrinsics (b,3,3)  相机内参
    #return (b,3,m,n), (b,3,m,n) x方向和y方向上的法向量
    dirs = OF.get_camera_pixel_directions(depth.shape[2:4], intrinsics, normalized_intrinsics=normalized_intrinsics) # 获取相机像素方向
    dirs = dirs.permute(0,3,1,2)
    if depth_is_along_ray:
        dirs = torch.nn.functional.normalize(dirs,dim=1)
    pc = dirs*depth # 将深度图乘以相机像素方向，得到点云
    
    normal = _point_cloud_to_normals(pc, diff_type=diff_type) # 计算法向量
    return normal, pc

class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dim=384, heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=heads, batch_first=True)  # Note the batch_first=True

    def forward(self, queries, keys_values):
        # Since we're ignoring class tokens, the input is directly [B, N, C]
        # Apply multi-head attention; assuming queries and keys_values are prepared [B, N, C]
        attn_output, attn_weights = self.attention(queries, keys_values, keys_values)

        # attn_output is already in the correct shape [B, N, C], so we return it directly
        return attn_output, attn_weights

class FeatureEncoder(nn.Module):
    """Encodes combined features into a lower-dimensional representation."""
    def __init__(self, input_channels, encoded_dim):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_channels // 2, encoded_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class FiLM(nn.Module):
    """Applies Feature-wise Linear Modulation to condition disparity refinement. 应用特征线性调制来调节视差细化"""
    def __init__(self, encoded_dim, target_channels):
        super(FiLM, self).__init__()
        self.scale_shift_net = nn.Linear(encoded_dim, target_channels * 2)

    def forward(self, features, disparity):
        # Global average pooling and processing to get scale and shift parameters 全局平均池化并处理以获得缩放和移动参数
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1) # 将特征展平
        scale_shift_params = self.scale_shift_net(pooled_features) # 将特征输入到线性层得到缩放和移动参数
        scale, shift = scale_shift_params.chunk(2, dim=1) # 将参数分成缩放和移动两部分
        
        scale = scale.unsqueeze(-1).unsqueeze(-1) # 将缩放参数扩展为与视差相同的形状
        shift = shift.unsqueeze(-1).unsqueeze(-1) # 将移动参数扩展为与视差相同的形状
        
        # Apply FiLM modulation to disparity
        modulated_disparity = disparity * scale + shift # 将视差乘以缩放参数并加上移动参数，得到调制后的视差
        return modulated_disparity # 返回调制后的视差

class PPSNet_Refinement(nn.Module):
    """Refines disparity map conditioned on encoded features."""

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self, disparity_channels, encoded_dim, feature_dim, heads=8):
        super(PPSNet_Refinement, self).__init__()
        # 用骨干的 token 维度 feature_dim (e.g., 1024 for vitl)
        self.encoder = FeatureEncoder(input_channels=feature_dim, encoded_dim=encoded_dim)
        self.film = FiLM(encoded_dim=encoded_dim, target_channels=disparity_channels)
        self.cross_attention = CrossAttentionModule(feature_dim=feature_dim, heads=heads)
        self.refinement_net = UNet(disparity_channels, disparity_channels)

        self.apply(self.init_weights)

    def forward(self, features_rgb, features_colored_dot_product, initial_disparity):
        # features_* 是由 DINOv2 get_intermediate_layers(..., return_class_token=True) 返回的列表
        # 每个元素是 (tokens, cls_token)，其中 tokens: [B, N, C]
        combined_attn = []
        for frgb, fcdp in zip(features_rgb, features_colored_dot_product):
            tokens_rgb = frgb[0]    # [B, N, C]
            tokens_cdp = fcdp[0]    # [B, N, C]
            attn_out, _ = self.cross_attention(tokens_rgb, tokens_cdp)  # [B, N, C]
            combined_attn.append(attn_out)

        # 取最后一层特征（与原代码一致用第4层，这里用 -1 更稳）
        attn_last = combined_attn[-1]         # [B, N, C]
        
        B, N, C = attn_last.shape
        # 从 N 推回 patch 网格大小：H_p * W_p = N；ViT/14 下通常 H_p=W_p
        Hp = Wp = int(N ** 0.5)
        assert Hp * Wp == N, f"Token count {N} 不是正方网格，无法 reshape；请检查输入尺寸与 patch 大小是否匹配"

        # [B, N, C] -> [B, C, Hp, Wp]
        feat_2d = attn_last.permute(0, 2, 1).reshape(B, C, Hp, Wp)

        # 编码得到条件向量
        encoded_features = self.encoder(feat_2d)  # -> [B, encoded_dim, Hp/2, Wp/2]

        # FiLM 调制
        modulated_disparity = self.film(encoded_features, initial_disparity)  # [B,1,H?,W?] 与 initial_disparity 对齐

        # U-Net 细化（与原逻辑一致）
        modulated_disparity = F.interpolate(modulated_disparity, scale_factor=0.5, mode='bilinear', align_corners=False)
        refined_disparity = self.refinement_net(modulated_disparity)

        # 这里不要写死 518×518，直接插回到 initial_disparity 的分辨率
        refined_disparity = F.interpolate(refined_disparity, size=initial_disparity.shape[-2:], mode='bilinear', align_corners=False)

        return (refined_disparity + initial_disparity).squeeze(1)

class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out
        
class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(DPT_DINOv2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        
        self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))

        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x, ref_dirs, light_pos, light_dir, mu, n_intrinsics):

        h, w = x.shape[-2:]
        
        # Step 1: Initial depth prediction and normals from depth 初始化深度预测和法线从深度
        features_rgb = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        disparity = self.depth_head(features_rgb, patch_h, patch_w) # 得到disparity
        disparity = F.interpolate(disparity, size=(h, w), mode="bilinear", align_corners=True) # 将disparity插值到原图大小
        disparity = F.relu(disparity) # 将disparity激活

        # Get depth from disparity 从disparity得到深度
        depth = 1 / disparity
        depth = torch.clamp(depth, 0, 1)

        normal, _ = _get_normals_from_depth(depth, n_intrinsics)

        # Step 2: Get PPL info from initial depth 从初始深度获取PPL信息
        pc_preds = depth.squeeze(1).unsqueeze(3)*ref_dirs # 深度图乘以相机像素方向，得到点云预测
        l, a = calculate_per_pixel_lighting(pc_preds, light_pos, light_dir, mu) # 计算PPL，输入点云预测、光源位置、光源方向、角衰减系数

        # Convert image to grayscale 将图像转换为灰度
        img_gray = _rgb_to_grayscale(x)  # Resulting shape: (B, 1, H, W)  rgb转灰度图

        # Ensure L and A are in the same format as RGB (B, C, H, W) 确保L和A与RGB格式相同（B，C，H，W）
        l = l.permute(0, 3, 1, 2)  # Rearrange l from [B, H, W, C] to [B, C, H, W] 将l从[B，H，W，C]重新排列为[B，C，H，W]
        a = a.permute(0, 3, 1, 2)  # Rearrange a from [B, H, W, C] to [B, C, H, W] 将a从[B，H，W，C]重新排列为[B，C，H，W]

        # Log transformation on A # 对A进行对数变换
        a = torch.log(a + 1e-8)

        # Min-Max normalization for A should ideally be based on dataset statistics, 归一化A的最小-最大值应基于数据集统计
        # Here we normalize based on the min and max of the current batch for simplicity # 这里我们根据当前批次的min和max进行简化归一化
        a_min, a_max = a.min(), a.max()
        a = (a - a_min) / (a_max - a_min + 1e-8)

        # Define the threshold for specular highlights # 定义镜面高光的阈值
        threshold = 0.98  # Adjust this threshold based on your data 根据您的数据调整此阈值
        # Create a mask for pixels above the intensity threshold  # 创建一个强度阈值以上的像素掩码
        specular_mask = (img_gray > threshold).float()  # [B, 1, H, W] 

        # Normalize l and normal
        l_norm = _normalize_vectors(l) # 归一化l
        normal_norm = _normalize_vectors(normal)  # 归一化normal

        # Compute dot product and apply attenuation # 计算点积并应用衰减
        dot_product = torch.sum(l_norm * normal_norm, dim=1, keepdim=True)  # l_norm Shape: (B, 3, H, W) normal_norm Shape: (B, 3, H, W) # 点积计算
        dot_product_clamped = torch.clamp(dot_product, -1, 1) # Shape: (B, 1, H, W) (L(x)点积N(x))，限制在-1到1之间
        dot_product_attenuated = dot_product_clamped * a # dot_product_clamped Shape: (B, 1, H, W) a Shape: (B, 1, H, W) # 点积乘以衰减因子a

        rgb_vis = x[0].permute(1, 2, 0).cpu().numpy() # Shape: (H, W, 3) # 将x[0]从(B, 3, H, W)转换为(H, W, 3)

        # Convert RGB to HSV
        hsv_vis = mcolors.rgb_to_hsv(rgb_vis) # Shape: (H, W, 3) # 将RGB转换为HSV

        # Retain H and S, but set V to 1.0 (max brightness) # 保留H和S，但将V设置为1.0（最大亮度）
        hsv_albedo = np.copy(hsv_vis) # Shape: (H, W, 3) # 复制HSV
        hsv_albedo[:, :, 2] = 1.0  # Set V to 100% brightness # 设置V为100%亮度

        # Convert back to RGB for visualization # 将其转换回RGB以进行可视化
        rgb_albedo_vis = mcolors.hsv_to_rgb(hsv_albedo) # Shape: (H, W, 3) # 将HSV转换回RGB

        h, w = x.shape[-2:] # 获取x的形状的最后一个维度的大小

        img_gray = img_gray.repeat(1, 3, 1, 1)  # New shape will be [B, 3, H, W] # 将img_gray重复3次以匹配通道数
        dot_product_attenuated = dot_product_attenuated.repeat(1, 3, 1, 1)  # New shape will be [B, 3, H, W] # 将dot_product_attenuated重复3次以匹配通道数
        
        # Assuming 'albedo_np' is your NumPy array containing the albedo image
        albedo_tensor = torch.from_numpy(rgb_albedo_vis).float() # 将albedo_np转换为PyTorch张量
        albedo_tensor = albedo_tensor.to('cuda').div(255.0) if albedo_tensor.max() > 1.0 else albedo_tensor.to('cuda') # 将albedo_tensor转换为GPU张量，并将其值归一化到[0, 1]范围内

        colored_dot_product_attenuated = albedo_tensor.permute(2, 0, 1).unsqueeze(0) * dot_product_attenuated # 将albedo_tensor的维度从[3, H, W]转换为[1, 3, H, W]，并将其与dot_product_attenuated相乘

        # Feature extraction for dot_product_attenuated
        features_colored_dot_product = self.pretrained.get_intermediate_layers(colored_dot_product_attenuated, 4, return_class_token=True) # 提取dot_product_attenuated的特征

        return disparity, features_rgb, features_colored_dot_product

class PPSNet_Backbone(DPT_DINOv2, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)


# if __name__ == "__main__":
    # img_tensor = torch.randn(1, 3, 224, 224).cuda()
    # ref_dirs = OF.get_camera_pixel_directions(img_tensor.shape[2:4], batch['n_intrinsics'], normalized_intrinsics=True).to(device)
    # light_data = [item.to(device) for item in batch['light_data']]
    # model = PPSNet_Backbone(config).cuda()
    # refinement_model = PPSNet_Refinement(1, 384)
    # disparity, rgb_feats, colored_dot_product_feats = model(img_tensor, ref_dirs, *light_data, batch['n_intrinsics'].to(device))
    # disp_preds = refinement_model(rgb_feats, colored_dot_product_feats, disparity)
    # print(disp_preds)
