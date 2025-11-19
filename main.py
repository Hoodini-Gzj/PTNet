# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from thop import profile
import argparse
import os
from functools import partial
# from monai.networks.nets import vista3d132
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from torchsummary import summary
from networks.unter import UNETR, UNETR_2c, UNETR_UniSeg
from networks.Promptunetr import UNETR_Prompt, UNETR_AVIT, UNETR_PGT
from networks.Promptunter_dualpath import UNETR_Prompt_dualpath,UNETR_Prompt_dualpath_av, UNETR_Prompt_dualpath_adapter, UNETR_Prompt_dualpath_adapter_dconv, UNETR_Prompt_dualpath_adapter_airway, \
    UNETR_Prompt_dualpath_dconv_airway, UNETR_Prompt_dualpath_airway, UNETR_Prompt_dualpath_dconv
# from networks.cas_net import CSNet3D
from networks.er_net import ER_Net
from networks.cs_net import CS_Net
from networks.pvsvas import pvsvasUNet3D
from networks.kiu_net import kiunet3d
from networks.kiu_net_1 import kiunet_min
from networks.custom_unter import UNETR_custom
from networks.nnformer import nnFormer_new, nnFormer
from networks.csa_net import CSNet3D
from networks.wings_net import WingsNet
from networks.unet3d import UNet3D, UNet3D_R1_ours
# from networks.vista3d import vista_model
from networks.Universal_model import Universal_model,Universal_model_ours,Universal_model_A

from networks.build_vista3d import vista_model_registry
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer_40_20 import run_training #batch size 1
# from trainer import run_training #batch size 2
# from trainer_dualpath import run_training #batch size 2

# from utils.data_utils_dualpath import get_loader
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="unetr_test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--pretrained_dir", default='F:/UNETR_R1/runs/unetr_test/', type=str, help="pretrained checkpoint directory")
parser.add_argument("--data_dir", default="./dataset/dataset1/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="carve14_20%_5_fold_v3.json", type=str, help="dataset json file")#dataset_vessel_5_fold_v1_dualpath
parser.add_argument("--pretrained_model_name", default='clip_driven_universal_swin_unetr.pth', type=str, help="pretrained model name")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=70, type=int, help="max number of training epochs")#总的训练epoch
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=50, type=int, help="validation frequency")#间隔多少val一次
parser.add_argument("--distributed", action="store_true", help="start distributed training")#并行训练？
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=0, type=int, help="number of workers")#多线程的？
parser.add_argument("--model_name", default="unetr", type=str, help="model name")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")##########################################################av是3 其他是2
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-950.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")  #########################
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")

# #train  https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV
# --feature_size=16
# --batch_size=1
# --logdir=unetr_test
# --optim_lr=1e-4
# --lrschedule=warmup_cosine
# --infer_overlap=0.5
# --save_checkpoint
# --data_dir=D:/Code/UNETR/dataset/dataset1
#fine-t
# --batch_size=1
# --logdir=unetr_test
# --optim_lr=1e-4
# --lrschedule=warmup_cosine
# --infer_overlap=0.5
# --save_checkpoint
# --data_dir=D:/Code/UNETR/dataset/dataset1/
# --pretrained_dir='D:/Code/UNETR/pretrained_models'
# --pretrained_model_name='model_final.pt'
# --resume_ckpt
def main():
    args = parser.parse_args()
    args.save_checkpoint = True
    args.amp = not args.noamp
    # args.logdir = "./runs/" + args.logdir
    args.logdir = "F:/UNETR_R1/runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
    if (args.model_name is None) or args.model_name == "unetr":
        # model = UNETR(
        #     in_channels=args.in_channels,
        #     out_channels=args.out_channels,
        #     img_size=(args.roi_x, args.roi_y, args.roi_z),
        #     feature_size=args.feature_size,
        #     hidden_size=args.hidden_size,
        #     mlp_dim=args.mlp_dim,
        #     num_heads=args.num_heads,
        #     pos_embed=args.pos_embed,
        #     norm_name=args.norm_name,
        #     conv_block=True,
        #     res_block=True,
        #     dropout_rate=args.dropout_rate,
        # )



        # model_Pre = UNETR_Prompt_dualpath_dconv_airway(
        #     in_channels=1,
        #     out_channels=2,
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed='perceptron',
        #     norm_name='instance',
        #     conv_block=True,
        #     res_block=True,
        #     dropout_rate=0.0)
        model_Pre = UNETR_2c(
            in_channels=1,
            out_channels=14,  # 原始的是14，我的是2 av是3或者4
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0)
        model = UNETR(
            in_channels=1,
            out_channels=2,  # 原始的是14，我的是2 av是3或者4
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0)
        # model_Pre = UNETR_2c(
        #     in_channels=1,
        #     out_channels=2,  # 原始的是14，我的是2 av是3或者4
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed='perceptron',
        #     norm_name='instance',
        #     conv_block=True,
        #     res_block=True,
        #     dropout_rate=0.0)
        # model = UNETR_AVIT(
        #     in_channels=1,
        #     out_channels=2,  # 原始的是14，我的是2
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed='perceptron',
        #     norm_name='instance',
        #     conv_block=False,
        #     res_block=False,
        #     dropout_rate=0.0)

        # model = UNETR_PromptMIL(
        #     in_channels=1,
        #     out_channels=2,  # 原始的是14，我的是2
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed='perceptron',
        #     norm_name='instance',
        #     conv_block=False,
        #     res_block=False,
        #     dropout_rate=0.0)
        # model = UNETR_UniSeg(
        #     in_channels=1,
        #     out_channels=2,  # 原始的是14，我的是2
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed='perceptron',
        #     norm_name='instance',
        #     conv_block=False,
        #     res_block=False,
        #     dropout_rate=0.0)
        # model = UNETR_PGT(
        #     in_channels=1,
        #     out_channels=2,
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed='perceptron',
        #     norm_name='instance',
        #     conv_block=True,
        #     res_block=True,
        #     dropout_rate=0.0)
        # model = UNETR_Prompt_dualpath(
        #     device=device,
        #     in_channels=1,
        #     out_channels=2,
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed='perceptron',
        #     norm_name='instance',
        #     conv_block=True,
        #     res_block=True,
        #     dropout_rate=0.0)
        # model = UNETR_Prompt_dualpath_dconv(
        #     device=device,
        #     in_channels=1,
        #     out_channels=2,
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed='perceptron',
        #     norm_name='instance',
        #     conv_block=True,
        #     res_block=True,
        #     dropout_rate=0.0)
        # model = UNETR_Prompt_dualpath_airway(
        #     device=device,
        #     in_channels=1,
        #     out_channels=2,
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed='perceptron',
        #     norm_name='instance',
        #     conv_block=True,
        #     res_block=True,
        #     dropout_rate=0.0)
        # model_Pre = UNETR_Prompt_dualpath_av(
        #     device=device,
        #     in_channels=1,
        #     out_channels=2,
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed='perceptron',
        #     norm_name='instance',
        #     conv_block=True,
        #     res_block=True,
        #     dropout_rate=0.0)
        #
        # model = UNETR_Prompt_dualpath_dconv_airway(
        #     device="cuda:0",
        #     in_channels=1,
        #     out_channels=2,
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed='perceptron',
        #     norm_name='instance',
        #     conv_block=True,
        #     res_block=True,
        #     dropout_rate=0.0)
        # model = model.to(device)




        # model = pvsvasUNet3D()
        # model_Pre = pvsvasUNet3D()
        # model = ER_Net()
        # model_Pre = ER_Net()
        # model = kiunet3d()
        # model = CS_Net()
        # model = kiunet_min()
        # model = UNETR_custom()
        # model = nnFormer_new()
        # model_Pre = nnFormer()
        # model = CSNet3D()
        # model_Pre = CS_Net()
        # model = WingsNet()
        # model_Pre = WingsNet()
        # model = UNet3D()
        # model_Pre = UNet3D()
        # model_Pre = Universal_model()
        # model = Universal_model_ours()
        # model = Universal_model_A()
        # model_Pre = vista_model()
        # model = vista_model()
        # model_Pre = vista_model_registry["vista3d_segresnet_vanilla"]()
        # model = vista_model_registry["vista3d_segresnet_ours"]()
        # model = vista_model_registry["vista3d_segresnet_A"]()
        # model = UNet3D_R1_ours()

        # # if args.resume_ckpt:
        # model_dict = torch.load(os.path.join('F:/UNETR_R1/runs/unetr_test/pv25/100%/ours_5_fold_v4.pt'))#carve14上微调
        # model.load_state_dict(model_dict.get('state_dict'))#carve14上微调
        # model_dict = torch.load(os.path.join('F:/UNETR_R1/runs/unetr_test/model_final.pt'))#carve14上微调
        # model.load_state_dict(model_dict.get('state_dict'), strict=False)#carve14上微调
            # Prompt用这个
        new_weights_dict = model.state_dict()#-------------------------------------------------Ours AViT
        Pre_weights_dict = torch.load(os.path.join(
            'D:/Code/UNETR/runs/unetr_test/UNETR_model_best_acc.pt'))  # unter预训练权重 model_final   UNETR_model_best_acc
        # Pre_weights_dict = torch.load(
        #     os.path.join('D:/Code/UNETR/runs/unetr_test/xiaorong dconv_airway.pt'))  # 附加实验-气管 a/v
        # model_Pre.load_state_dict(Pre_weights_dict.get('state_dict'))#我的方法
        model_Pre.load_state_dict(Pre_weights_dict.state_dict(), strict=False)  # unter预训练权重
        Pre_weights_dict = model_Pre.state_dict()
        for k in Pre_weights_dict.keys():
            if k in new_weights_dict.keys():
                print(f"Loading {k}, shape {Pre_weights_dict[k].shape}")
                new_weights_dict[k] = Pre_weights_dict[k]

        # 手动判断 missing 和 unexpected
        missing_keys = [k for k in new_weights_dict.keys() if k not in Pre_weights_dict]
        unexpected_keys = [k for k in Pre_weights_dict.keys() if k not in new_weights_dict]

        print("Missing keys in pretrained weights:", missing_keys)
        print("Unexpected keys in pretrained weights:", unexpected_keys)

        # 最后加载
        model.load_state_dict(new_weights_dict)#-------------------------------------------------Ours AViT
        #
        # new_weights_dict = model.state_dict()  # -------------------------------------------------PGT
        # Pre_weights_dict = torch.load(os.path.join(
        #     'D:/Code/UNETR/runs/unetr_test/UNETR_model_best_acc.pt'))  # unter预训练权重 model_final   UNETR_model_best_acc
        #
        # model_Pre.load_state_dict(Pre_weights_dict.state_dict(), strict=False)  # unter预训练权重
        # Pre_weights_dict = model_Pre.state_dict()
        # for k in Pre_weights_dict.keys():
        #     if k in new_weights_dict.keys():
        #         if Pre_weights_dict[k].shape == new_weights_dict[k].shape:
        #             print(f"Loading {k}, shape {Pre_weights_dict[k].shape}")
        #             new_weights_dict[k] = Pre_weights_dict[k]
        #
        # # 手动判断 missing 和 unexpected
        # missing_keys = [k for k in new_weights_dict.keys() if k not in Pre_weights_dict]
        # unexpected_keys = [k for k in Pre_weights_dict.keys() if k not in new_weights_dict]
        #
        # print("Missing keys in pretrained weights:", missing_keys)
        # print("Unexpected keys in pretrained weights:", unexpected_keys)
        #
        # # 最后加载
        # model.load_state_dict(new_weights_dict)  # -------------------------------------------------PGT
        #

        # ckpt = torch.load("F:/UNETR_R1/runs/unetr_test/synapse_pretrain.model", map_location="cpu")
        #
        # state_dict = {}
        # for k, v in ckpt.items():
        #     # 跳过相对位置编码参数
        #     if "relative_position_bias_table" in k or "relative_position_index" in k:
        #         print(f"Skip {k}")
        #         continue
        #     state_dict[k] = v
        #
        # missing, unexpected = model.load_state_dict(ckpt, strict=False)
        # print("Missing keys:", missing)
        # print("Unexpected keys:", unexpected)
        # print("Unexpected keys:", unexpected)  # -------------------------------------------------

        # ckpt = torch.load("F:/UNETR_R1/runs/unetr_test/synapse_pretrain.model", map_location="cpu")#-------------------------------------------------nnformer R1的问题，我们方法在其他模型上的表现
        #
        # model_dict = model.state_dict()
        # load_dict = {}
        #
        # missing_keys = []
        # unexpected_keys = []
        # shape_mismatch = []
        #
        # for k, v in ckpt.items():
        #     if k not in model_dict:
        #         unexpected_keys.append(k)
        #         continue
        #     # if "relative_position_bias_table" in k or "relative_position_index" in k:
        #     #     print(f"Skip {k} (relative position)")
        #     #     continue
        #     if v.shape != model_dict[k].shape:
        #         shape_mismatch.append((k, v.shape, model_dict[k].shape))
        #         continue
        #     load_dict[k] = v
        #
        # # 更新模型
        # model_dict.update(load_dict)
        # model.load_state_dict(model_dict)
        #
        # # 统计 missing keys
        # for k in model.state_dict().keys():
        #     if k not in load_dict:
        #         missing_keys.append(k)
        #
        # print("=" * 50)
        # print(f"Loaded params: {len(load_dict)} / {len(model.state_dict())}")
        # print("Missing keys:", missing_keys)
        # print("Unexpected keys:", unexpected_keys)
        # print("Shape mismatch keys:")
        # for k, ckpt_shape, model_shape in shape_mismatch:
        #     print(f"  {k}: ckpt {tuple(ckpt_shape)} vs model {tuple(model_shape)}")
        # print("=" * 50)#-------------------------------------------------nnformer R1的问题，我们方法在其他模型上的表现

        # checkpoint = torch.load(  #只能测试的时候用了（计算复杂度） #-------------------------------------------------Universal_model(unet) R1的问题，我们方法在其他模型上的表现
        #     "F:/UNETR_R1/runs/unetr_test/clip_driven_universal_unet.pth",#clip_driven_universal_swin_unetr clip_driven_universal_unet
        #     map_location="cpu"
        # )
        #
        # state_dict = checkpoint["net"]  # 如果保存时就是 {'net': model.state_dict()}
        # # 去掉 DataParallel 保存时的 'module.' 前缀
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith("module."):
        #         new_state_dict[k[len("module."):]] = v
        #     else:
        #         new_state_dict[k] = v
        # if "organ_embedding" in new_state_dict:
        #     del new_state_dict["organ_embedding"]
        # missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        #
        # print("Missing keys:", missing)
        # print("Unexpected keys:", unexpected)#-------------------------------------------------Universal_model(unet)R1的问题，我们方法在其他模型上的表现

        # checkpoint = torch.load(#-------------------------------------------------Universal_model(swinUNTER)
        #     "F:/UNETR_R1/runs/unetr_test/clip_driven_universal_swin_unetr.pth",
        #     # clip_driven_universal_swin_unetr clip_driven_universal_unet
        #     map_location="cpu"
        # )
        #
        # state_dict = checkpoint["net"]  # 如果保存时就是 {'net': model.state_dict()}
        # # 去掉 DataParallel 保存时的 'module.' 前缀
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith("module."):
        #         k = k[len("module."):]
        #     # 给所有 key 加 "backbone." 前缀
        #     if not k.startswith("backbone."):
        #         k = "backbone." + k
        #     new_state_dict[k] = v
        # missing, unexpected = model.load_state_dict(new_state_dict,
        #                                             strict=False)  # -------------------------------------------------Universal_model(swinUNTER)
        #
        # print("Missing keys:", missing)
        # print("Unexpected keys:", unexpected)  # -------------------------------------------------Universal_model(swinUNTER)



        # print("Missing keys:", missing)
        # print("Unexpected keys:", unexpected)
        # print("Unexpected keys:", unexpected)
        # Pre_weights_dict = model_Pre.state_dict()
        # for k in Pre_weights_dict.keys():
        #     # print(k)
        #     if k in new_weights_dict.keys():
        #         print(k)
        #         new_weights_dict[k] = Pre_weights_dict[k]
        # model.load_state_dict(new_weights_dict)#-------------------------------------------------Universal_model

        # new_weights_dict = model.state_dict()  # -------------------------------------------------Universal_model
        #
        # checkpoint = torch.load(  # 只能测试的时候用了（计算复杂度）
        #     "F:/UNETR_R1/runs/unetr_test/clip_driven_universal_swin_unetr.pth",
        #     map_location="cpu"
        # )
        #
        # state_dict = checkpoint["net"]  # 如果保存时就是 {'net': model.state_dict()}
        #
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     # 去掉多余的 "module."
        #     if k.startswith("module."):
        #         k = k[len("module."):]
        #     # 补上 "backbone." 前缀
        #     if k.startswith("swinViT"):
        #         k = "backbone." + k
        #     if k.startswith("encoder") or k.startswith("decoder"):
        #         k = "backbone." + k
        #     new_state_dict[k] = v
        #
        # # 加载到模型
        # missing, unexpected = model_Pre.load_state_dict(new_state_dict, strict=True)
        #
        # print("Missing keys:", missing)
        # print("Unexpected keys:", unexpected)
        #
        # Pre_weights_dict = model_Pre.state_dict()
        # for k in Pre_weights_dict.keys():
        #     # print(k)
        #     if k in new_weights_dict.keys():
        #         print(k)
        #         new_weights_dict[k] = Pre_weights_dict[k]
        # model.load_state_dict(new_weights_dict)  # -------------------------------------------------Universal_model


        # # 遍历预训练模型的权重，检查尺寸是否匹配并跳过 vit.patch_embedding
        # for k in Pre_weights_dict.keys():
        #     if k in new_weights_dict:
        #         # 如果是 vit.patch_embedding 或 vit.patch_embedding 下的层则跳过
        #         if 'vit.patch_embedding' in k:
        #             print(f"Skipping weight for: {k} (patch embedding layer)")
        #         elif Pre_weights_dict[k].size() == new_weights_dict[k].size():
        #             print(f"Loading weight for: {k}")
        #             new_weights_dict[k] = Pre_weights_dict[k]  # 如果尺寸匹配则加载
        #         else:
        #             print(
        #                 f"Skipping weight for: {k}, size mismatch {Pre_weights_dict[k].size()} vs {new_weights_dict[k].size()}")
        #     else:
        #         print(f"Skipping weight for: {k}, key not found in new model.")
        #
        # # 将更新后的权重加载到模型中
        # model.load_state_dict(new_weights_dict)


            # 正常预训练用这个
        # model_dict = torch.load(os.path.join('F:/UNETR_R1/runs/unetr_test/model_final.pt'))
        # model.load_state_dict(model_dict.get('state_dict'))
        # print("Use pretrained weights")

        # # 1. 获取当前模型的 state_dict#-----------------------------vista3d R1的问题，我们方法在其他模型上的表现
        # new_weights_dict = model.state_dict()
        #
        # # 2. 加载预训练权重
        # Pre_weights_dict = torch.load(
        #     os.path.join('F:/UNETR_R1/runs/unetr_test/vista3d_model.pt')
        # )
        #
        # # 3. 只把形状匹配的权重覆盖到 new_weights_dict
        # for k, v in Pre_weights_dict.items():
        #     if k in new_weights_dict:
        #         if v.shape == new_weights_dict[k].shape:
        #             print(f"Loading {k}, shape {v.shape}")
        #             new_weights_dict[k] = v
        #         else:
        #             print(f"Skip {k}, pretrained {v.shape}, current {new_weights_dict[k].shape}")
        #
        # # 4. 加载到模型
        # missing, unexpected = model.load_state_dict(new_weights_dict, strict=False)
        # print("Missing keys:", missing)
        # print("Unexpected keys:", unexpected)#-----------------------------vista3d R1的问题，我们方法在其他模型上的表现

        # model_dict = torch.load(os.path.join('F:/UNETR_R1/runs/unetr_test/model_final.pt'))
        # model.load_state_dict(model_dict.get('state_dict'))
        # print("Use pretrained weights")

        # # 1. 获取当前模型的 state_dict#-----------------------------vista3d
        # new_weights_dict = model.state_dict()
        #
        # # 2. 加载预训练权重
        # Pre_weights_dict = torch.load(
        #     os.path.join('F:/UNETR_R1/runs/unetr_test/clip_driven_universal_swin_unetr.pth')
        # )
        #
        # # 3. 只把形状匹配的权重覆盖到 new_weights_dict
        # for k, v in Pre_weights_dict.items():
        #     if k in new_weights_dict:
        #         if v.shape == new_weights_dict[k].shape:
        #             print(f"Loading {k}, shape {v.shapfe}")
        #             new_weights_dict[k] = v
        #         else:
        #             print(f"Skip {k}, pretrained {v.shape}, current {new_weights_dict[k].shape}")
        #
        # # 4. 加载到模型
        # missing, unexpected = model.load_state_dict(new_weights_dict, strict=True)
        # print("Missing keys:", missing)
        # print("Unexpected keys:", unexpected)#-----------------------------vista3d

        input = torch.randn(1, 1, 96, 96, 96)  # 输入尺寸要与你的任务一致

        flops, params = profile(model, inputs=(input,))

        print(f"FLOPs: {flops / 1e9:.2f} G")
        print(f"Params: {params / 1e6:.2f} M")
        if args.resume_jit:
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    dice_loss = DiceCELoss(
        include_background=True, to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr#9.30include_background改成了False
    )
    # post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels) #monai=0.7 ValueError: `to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.
    post_label = AsDiscrete(to_onehot=None, n_classes=args.out_channels) #monai=1.3.2
    post_pred = AsDiscrete(argmax=True, to_onehot=None, n_classes=args.out_channels)#同上
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)#9.30include_background改成了False
    model_inferer = partial(#最初的版本
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    # model_inferer = None#unetr_dual_path 2024-3-27

    # pytorch_total_params = sum(p.numel() for p in model_Pre.parameters() if p.requires_grad)
    # print("Total parameters count model_Pre", pytorch_total_params)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    # params = []#############################################################################prompt,下面的optimizer的model.parameters()要改成params
    # train_layer = ['PromptBlock']
    # for name, param in model.named_parameters():
    #     if any(name.startswith(prefix) for prefix in train_layer):
    #         print(name)
    #         params.append(param)
    #     else:
    #         param.requires_gard = False


    # vit = 11620390
    # mo = 7171140

    # for name, param in model.named_parameters():#-------------------------------------------------nnformer R1的问题，我们方法在其他模型上的表现
    #     if 'relative_position_bias_table' not in name and 'relative_position_index' not in name and 'final' not in name and 'decoder' not in name and 'patch_embed' not in name and 'adapter' not in name:
    #         # print(name)
    #         param.requires_grad = False#-------------------------------------------------nnformer R1的问题，我们方法在其他模型上的表现

    # for name, param in model.named_parameters():#-------------------------------------------------Universal_model(Unet) R1的问题，我们方法在其他模型上的表现
    #     if 'up_tr256' not in name and 'up_tr128' not in name and 'up_tr64' not in name and 'precls_conv' not in name and 'GAP' not in name and 'controller' not in name and 'organ_embedding' not in name and 'adapter' not in name and 'down_tr64' not in name and 'text_to_vision' not in name:
    #         # print(name)
    #         param.requires_grad = False#-------------------------------------------------Universal_model(Unet) R1的问题，我们方法在其他模型上的表现
    # for name, param in model.named_parameters():#-------------------------------------------------Universal_model(SwinUNETR) R1的问题，我们方法在其他模型上的表现
    #     if 'decoder' not in name and 'precls_conv' not in name and 'GAP' not in name and 'controller' not in name and 'organ_embedding' not in name and 'adapter' not in name and 'encoder' not in name:
    #         # print(name)
    #         param.requires_grad = False#-------------------------------------------------Universal_model(SwinUNETR) R1的问题，我们方法在其他模型上的表现

    # for name, param in model.named_parameters():#-------------------------------------------------vista3d R1的问题，我们方法在其他模型上的表现
    #     if 'up_layers' not in name and 'up_layers_auto' not in name and 'class_head' not in name and 'point_head' not in name and 'weight_mapper' not in name and 'adapter' not in name and 'airway' not in name:
    #         # print(name)
    #         param.requires_grad = False#-------------------------------------------------vista3d R1的问题，我们方法在其他模型上的表现


    # for name, param in model.named_parameters():#and 'airway' not in name是训练原始的，dualpath要去掉
    #     if 'adapter' not in name and 'norm' not in name and 'decoder' not in name and 'out' not in name and 'bias' not in name and 'airway' not in name:
    #         # print(name)
    #         param.requires_grad = False

    # for name, param in model.named_parameters():#and 'airway' not in name是训练原始的，dualpath要去掉
    #     if 'adapter' not in name and 'norm' not in name and 'decoder' not in name and 'out' not in name and 'bias' not in name:
    #         # print(name)
    #         param.requires_grad = False

    #
    # for name, param in model.named_parameters():#dualpath的
    #     if 'airway' in name:
    #         if 'adapter' in name or 'decoder' in name:
    #             param.requires_grad = False
    #     elif 'adapter' not in name and 'norm' not in name and 'decoder' not in name and 'out' not in name and 'bias' not in name:
    #         param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if 'adapter' not in name and 'decoder' not in name and 'encoder' not in name:
    #         param.requires_grad = False

    # for name, param in model.named_parameters():#AVIT
    #     if 'adapter' not in name and 'decoder' not in name and 'out' not in name:
    #         param.requires_grad = False

    # for name, param in model.named_parameters():#PGT
    #     if 'decoder' not in name and 'encoder' not in name and 'adapter' not in name and 'out' not in name:
    #         param.requires_grad = False

    # for name, param in model.named_parameters():#2c
    #     if 'out_2c' not in name:
    #         param.requires_grad = False


    # for name, param in model.named_parameters():#dualpath的
    #     if 'outputs' in name:
    #         if 'vit' in name:
    #             param.requires_grad = False
    #         else:
    #             param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    # for name, param in model.named_parameters():
    #     if 'dc0_0' not in name and 'dc0_1' not in name:
    #         param.requires_grad = False


    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # for name, param in model.named_parameters():
    #     if 'adapter' in name or ('vit' in name and 'adapter' in name):
    #         param.requires_grad = True
    #     elif 'norm' in name and ('vit' in name and 'norm' in name):
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    # 打印已设置requires_grad=True的层

    # for name, param in model.named_parameters():#消融实验 只有adapter
    #     if 'adapter' not in name and 'norm' not in name and 'decoder' not in name and 'out' not in name and 'bias' not in name:
    #             # print(name)
    #             param.requires_grad = False



    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Parameters: {param.numel() / 1e6} M")

    param = sum(p.numel() for p in model.parameters())
    print(f"number of parameter: {param/1e6} M")
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable parameter: {param/1e6} M")
    print(f"number of trainable parameter: {param / 1e6} M")
    # for n, value in model.parameters():  # PromptBlock
    #     if "PromptBlock" not in n:
    #         value.requires_grad = False
    #
    # model.parameters()


    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        test_loader=loader[2],
        # add_loader=loader[3]
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
