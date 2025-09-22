from __future__ import absolute_import, division, print_function

import os
import argparse
import time

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="IID_SFM options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory")

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default=time.strftime('%Y-%m-%d-%H-%M-%S'))
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["endovis", "hamlyn", "c3vd"],   # <-- add c3vd to be tolerant
                                 default="endovis")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="endovis",
                                 choices=["endovis", "hamlyn", "c3vd"])  # <-- add c3vd
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=320)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0.01)
        self.parser.add_argument("--reconstruction_constraint",
                                 type=float,
                                 help="consistency constraint weight",
                                 default=0.2)
        self.parser.add_argument("--gt_depth_weight",
                                type=float,
                                default=1.0,
                                help="lambda for supervised GT depth loss")
        self.parser.add_argument("--depth_consistency",
                                 type=float,
                                 help="depth consistency constraint weight",
                                default=0.1)        
        self.parser.add_argument("--reflec_constraint",
                                 type=float,
                                 help="epipolar constraint weight",
                                 default=0.2)
        self.parser.add_argument("--pps_pretrained_ckpt",
                                type=str,
                                default=None,
                                help="path to PPSNet checkpoint (.pth) with keys: student_state_dict/refiner_state_dict")
        self.parser.add_argument("--freeze_pps_backbone",
                                action="store_true",
                                default=False,
                                help="freeze PPSNet backbone")
        self.parser.add_argument("--pps_encoder",
                                type=str,
                                default="vits",  # vits/vitb/vitl
                                choices=["vits","vitb","vitl"],
                                help="DINOv2 encoder size for PPSNet")

        self.parser.add_argument("--reprojection_constraint",
                                 type=float,
                                 help="geometry constraint weight",
                                 default=0.1)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=1e-6)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=1.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # -------- C3VD specific (new) --------
        self.parser.add_argument("--c3vd_train_list",
                                 type=str,
                                 help="path to C3VD train list file (e.g. datasets/C3VD_splits/train.txt)")
        self.parser.add_argument("--c3vd_val_list",
                                 type=str,
                                 help="path to C3VD val list file (e.g. datasets/C3VD_splits/val.txt)")

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=1)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=30)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=10)

        # ABLATION options
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load"
                                 )
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load")

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=200)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true",
                                 default=True)
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="endovis",
                                 choices=["endovis", "hamlyn", "c3vd"],  # <-- add c3vd
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        opts = self.parser.parse_args()

        # ------- Post-processing for C3VD -------
        if opts.dataset.lower() == "c3vd":
            # 若用户未显式改动，单帧数据用 frame_ids=[0]
            # （你把 C3VD_Dataset 扩展成序列后，可直接在命令行传 -1 0 1 来覆盖）
            if opts.frame_ids == [0, -1, 1]:
                opts.frame_ids = [0]

            # 确保输入分辨率是 32 的倍数（monodepth2 解码器更稳）
            new_h = (opts.height // 32) * 32
            new_w = (opts.width // 32) * 32
            if new_h != opts.height or new_w != opts.width:
                print(f"[INFO] Snap C3VD input size from ({opts.height},{opts.width}) "
                      f"to nearest multiples of 32 -> ({new_h},{new_w})")
                opts.height, opts.width = new_h, new_w

            # split 在 C3VD 下不参与实际构建，允许保留默认值以兼容日志/评估路径

            # c3vd_train_list / c3vd_val_list 若没传，trainer 会使用默认:
            # datasets/C3VD_splits/{train,val}.txt

        self.options = opts
        return self.options
