# EDSR baseline model (x2) + JPEG augmentation
# python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble

# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results

# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt

# CUDA_VISIBLE_DEVICES=0 python -u main.py --model CGSRN --scale 4 --save CGSRN_BIX4_G2R24 --n_resgroups 2 --n_resblocks 24 --n_feats 128 --res_scale 0.1 --direct_up False --reset --data_test Set5+Set14 --batch_size 7 --patch_size 192 --save_results --lr 0.0001 --decay 150-300 --epochs 0 &

# CUDA_VISIBLE_DEVICES=0 python -u main.py --model CGSRN --scale 4 --save CGSRN_BIX4_G2R5 --n_resgroups 2 --n_resblocks 5 --n_feats 128 --res_scale 0.1 --direct_up False --reset --data_test Set5+Set14 --batch_size 8 --patch_size 192 --save_results --lr 0.0001 --decay 200-400 --epochs 0 &

# CUDA_VISIBLE_DEVICES=0 python -u main.py --model CGSRN --scale 4 --save CGSRN_BIX4_G2R2M5V1 --n_resgroups 2 --n_resblocks 2 --n_feats 128 --reset --data_test Set5+Set14 --batch_size 5 --patch_size 192 --save_results --lr 0.0001 --decay 150-300-450 --epochs 0 --test_every 0 --version v1


# test EDSR (EDSR-PyTorch)
# CUDA_VISIBLE_DEVICES=2 python -u main.py --model EDSR --data_test Set5+Set14+B100+Urban100+Manga109 --save EDSR_BIX2 --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train /home/yamengxi/models/EDSR/EDSR_x2.pt --test_only --save_results
# CUDA_VISIBLE_DEVICES=2 python -u main.py --model EDSR --data_test Set5+Set14+B100+Urban100+Manga109 --save EDSR_BIX3 --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train /home/yamengxi/models/EDSR/EDSR_x3.pt --test_only --save_results
# CUDA_VISIBLE_DEVICES=2 python -u main.py --model EDSR --data_test Set5+Set14+B100+Urban100+Manga109 --save EDSR_BIX4 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train /home/yamengxi/models/EDSR/EDSR_x4.pt --test_only --save_results

# test DBPN (DBPN-PyTorch)
# CUDA_VISIBLE_DEVICES=2 python -u eval.py --upscale_factor 2 --input_dir /home/yamengxi/datasets_for_EDSR-PyTorch/benchmark/ --test_dataset B100/LR_bicubic/X2     --model_type DBPN --model /home/yamengxi/DBPN-Pytorch/models/DBPN_x2.pth
# CUDA_VISIBLE_DEVICES=2 python -u eval.py --upscale_factor 2 --input_dir /home/yamengxi/datasets_for_EDSR-PyTorch/benchmark/ --test_dataset Manga109/LR_bicubic/X2 --model_type DBPN --model /home/yamengxi/DBPN-Pytorch/models/DBPN_x2.pth
# CUDA_VISIBLE_DEVICES=2 python -u eval.py --upscale_factor 2 --input_dir /home/yamengxi/datasets_for_EDSR-PyTorch/benchmark/ --test_dataset Set5/LR_bicubic/X2     --model_type DBPN --model /home/yamengxi/DBPN-Pytorch/models/DBPN_x2.pth
# CUDA_VISIBLE_DEVICES=2 python -u eval.py --upscale_factor 2 --input_dir /home/yamengxi/datasets_for_EDSR-PyTorch/benchmark/ --test_dataset Set14/LR_bicubic/X2    --model_type DBPN --model /home/yamengxi/DBPN-Pytorch/models/DBPN_x2.pth
# CUDA_VISIBLE_DEVICES=2 python -u eval.py --upscale_factor 2 --input_dir /home/yamengxi/datasets_for_EDSR-PyTorch/benchmark/ --test_dataset Urban100/LR_bicubic/X2 --model_type DBPN --model /home/yamengxi/DBPN-Pytorch/models/DBPN_x2.pth
# CUDA_VISIBLE_DEVICES=2 python -u eval.py --upscale_factor 4 --input_dir /home/yamengxi/datasets_for_EDSR-PyTorch/benchmark/ --test_dataset B100/LR_bicubic/X4     --model_type DBPN --model /home/yamengxi/DBPN-Pytorch/models/DBPN_x4.pth
# CUDA_VISIBLE_DEVICES=2 python -u eval.py --upscale_factor 4 --input_dir /home/yamengxi/datasets_for_EDSR-PyTorch/benchmark/ --test_dataset Manga109/LR_bicubic/X4 --model_type DBPN --model /home/yamengxi/DBPN-Pytorch/models/DBPN_x4.pth
# CUDA_VISIBLE_DEVICES=2 python -u eval.py --upscale_factor 4 --input_dir /home/yamengxi/datasets_for_EDSR-PyTorch/benchmark/ --test_dataset Set5/LR_bicubic/X4     --model_type DBPN --model /home/yamengxi/DBPN-Pytorch/models/DBPN_x4.pth
# CUDA_VISIBLE_DEVICES=2 python -u eval.py --upscale_factor 4 --input_dir /home/yamengxi/datasets_for_EDSR-PyTorch/benchmark/ --test_dataset Set14/LR_bicubic/X4    --model_type DBPN --model /home/yamengxi/DBPN-Pytorch/models/DBPN_x4.pth
# CUDA_VISIBLE_DEVICES=2 python -u eval.py --upscale_factor 4 --input_dir /home/yamengxi/datasets_for_EDSR-PyTorch/benchmark/ --test_dataset Urban100/LR_bicubic/X4 --model_type DBPN --model /home/yamengxi/DBPN-Pytorch/models/DBPN_x4.pth

# test RCAN (EDSR-PyTorch)
# CUDA_VISIBLE_DEVICES=2 python -u main.py --template RCAN --save RCAN_BIX2 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2 --pre_train /home/yamengxi/models/RCAN/RCAN_BIX2.pt --test_only --save_results
# CUDA_VISIBLE_DEVICES=2 python -u main.py --template RCAN --save RCAN_BIX3 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 3 --pre_train /home/yamengxi/models/RCAN/RCAN_BIX3.pt --test_only --save_results
# CUDA_VISIBLE_DEVICES=2 python -u main.py --template RCAN --save RCAN_BIX4 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 4 --pre_train /home/yamengxi/models/RCAN/RCAN_BIX4.pt --test_only --save_results


# test RDN (no)
# CUDA_VISIBLE_DEVICES=2 python -u main.py --model RDN --data_test Set5+Set14+B100+Urban100+Manga109 --save RDN_BIX4 --scale 4 --pre_train /home/yamengxi/models/RDN/RDN_BIX4.t7 --test_only --save_results

# test SAN (EDSR-PyTorch)
# CUDA_VISIBLE_DEVICES=2 python -u main.py --model SAN --save SAN_BIX2 --scale 2 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --data_test Set5+B100+Urban100+Manga109 --pre_train /home/yamengxi/models/SAN/SAN_BI2X.pt --test_only --save_results
# CUDA_VISIBLE_DEVICES=2 python -u main.py --model SAN --save SAN_BIX2 --scale 2 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --data_test Set14 --pre_train /home/yamengxi/models/SAN/SAN_BI2X.pt --test_only --save_results
# CUDA_VISIBLE_DEVICES=2 python -u main.py --model SAN --save SAN_BIX4 --scale 4 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --data_test Set5+Set14+B100+Urban100+Manga109 --pre_train /home/yamengxi/models/SAN/SAN_BI4X.pt --test_only --save_results

# test HAN (EDSR-PyTorch)
# CUDA_VISIBLE_DEVICES=2 python -u main.py --model HAN --n_resblocks 20 --n_feats 64 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2 --pre_train /home/yamengxi/models/HAN/HAN_BIX2.pt --test_only --save HAN_BIX2 --save_results
# CUDA_VISIBLE_DEVICES=2 python -u main.py --model HAN --n_resblocks 20 --n_feats 64 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 3 --pre_train /home/yamengxi/models/HAN/HAN_BIX3.pt --test_only --save HAN_BIX3 --save_results
# CUDA_VISIBLE_DEVICES=2 python -u main.py --model HAN --n_resblocks 20 --n_feats 64 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 4 --pre_train /home/yamengxi/models/HAN/HAN_BIX4.pt --test_only --save HAN_BIX4 --save_results



# train bfn
# CUDA_VISIBLE_DEVICES=0,1 python -u main.py --n_GPUs 2 \
# --model BFN --scale 2 --n_feats 64 --n_resblocks 12 --act prelu --main_block_version v1 --butterfly_conv_version v1 \
# --save BFN_BIX2_F64R12_MBV1_BFCV1 --reset --data_test Set5+Set14+B100+Urban100 --batch_size 36 --patch_size 128 --save_results --lr 0.0004 --decay 200-400-600-800-1000 --epochs 0 # --test_every 0

# CUDA_VISIBLE_DEVICES=0 python -u main.py \
# --model BFN --scale 2 --n_feats 64 --n_resblocks 12 --act prelu --main_block_version v1 --butterfly_conv_version v2 \
# --save BFN_BIX2_F64R12_MBV1_BFCV2 --reset --data_test Set5+Set14+B100+Urban100 --batch_size 20 --patch_size 128 --save_results --lr 0.0004 --decay 200-400-600-800-1000 --epochs 0 # --test_every 0

# CUDA_VISIBLE_DEVICES=0 python -u main.py \
# --model BFN --scale 2 --n_feats 64 --n_resblocks 12 --act prelu --main_block_version v1 --butterfly_conv_version v3 \
# --save BFN_BIX2_F64R12_MBV1_BFCV3 --reset --data_test Set5+Set14+B100+Urban100 --batch_size 20 --patch_size 128 --save_results --lr 0.0004 --decay 200-400-600-800-1000 --epochs 0 # --test_every 0

# CUDA_VISIBLE_DEVICES=0,1 python -u main.py --n_GPUs 2 \
# --model BFN --scale 2 --n_feats 64 --n_resblocks 9 --act prelu --main_block_version v2 --butterfly_conv_version v1 \
# --save BFN_BIX2_F64R11_MBV2_BFCV1 --reset --data_test Set5+Set14+B100+Urban100 --batch_size 20 --patch_size 128 --save_results --lr 0.0004 --decay 200-400-600-800-1000 --epochs 0 # --test_every 0

# CUDA_VISIBLE_DEVICES=0 python -u main.py \
# --model BFN --scale 2 --n_feats 64 --n_resblocks 6 --act prelu --main_block_version v2 --butterfly_conv_version v2 \
# --save BFN_BIX2_F64R12_MBV2_BFCV2 --reset --data_test Set5+Set14+B100+Urban100 --batch_size 10 --patch_size 128 --save_results --lr 0.0004 --decay 200-400-600-800-1000 --epochs 0 # --test_every 0

# CUDA_VISIBLE_DEVICES=0 python -u main.py \
# --model BFN --scale 2 --n_feats 64 --n_resblocks 6 --act prelu --main_block_version v2 --butterfly_conv_version v3 \
# --save BFN_BIX2_F64R12_MBV2_BFCV3 --reset --data_test Set5+Set14+B100+Urban100 --batch_size 10 --patch_size 128 --save_results --lr 0.0004 --decay 200-400-600-800-1000 --epochs 0 # --test_every 0

# CUDA_VISIBLE_DEVICES=0 python -u main.py \
# --model BFN --scale 2 --n_feats 64 --n_resblocks 5 --act prelu --main_block_version v3 --butterfly_conv_version v1 \
# --save BFN_BIX2_F64R5_MBV3_BFCV1 --reset --data_test Set5+Set14+B100+Urban100 --batch_size 10 --patch_size 128 --save_results --lr 0.0004 --decay 200-400-600-800-1000 --epochs 0 # --test_every 0

# CUDA_VISIBLE_DEVICES=0 python -u main.py \
# --model BFN --scale 2 --n_feats 64 --n_resblocks 5 --act prelu --main_block_version v3 --butterfly_conv_version v2 \
# --save BFN_BIX2_F64R5_MBV3_BFCV2 --reset --data_test Set5+Set14+B100+Urban100 --batch_size 10 --patch_size 128 --save_results --lr 0.0004 --decay 200-400-600-800-1000 --epochs 0 # --test_every 0

# CUDA_VISIBLE_DEVICES=0 python -u main.py \
# --model BFN --scale 2 --n_feats 64 --n_resblocks 5 --act prelu --main_block_version v3 --butterfly_conv_version v3 \
# --save BFN_BIX2_F64R5_MBV3_BFCV3 --reset --data_test Set5+Set14+B100+Urban100 --batch_size 10 --patch_size 128 --save_results --lr 0.0004 --decay 200-400-600-800-1000 --epochs 0 # --test_every 0


# train RFDN
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u main.py --n_GPUs 6 \
# --model RFDN --scale 2 --n_feats 48 --n_resblocks 6 --act lrelu --basic_module_version v1 \
# --save RFDN_BIX2_F48R6_BMV1 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u main.py --n_GPUs 6 \
# --model RFDN --scale 2 --n_feats 64 --n_resblocks 8 --act prelu --basic_module_version v3 \
# --save RFDN_BIX2_F64R8_BMV3 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u main.py --n_GPUs 6 \
# --model RFDN --scale 2 --n_feats 64 --n_resblocks 8 --act prelu --basic_module_version v4 \
# --save RFDN_BIX2_F64R8_BMV4 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u main.py --n_GPUs 6 \
# --model RFDN --scale 2 --n_feats 64 --n_resblocks 8 --act prelu --basic_module_version v5 \
# --save RFDN_BIX2_F64R8_BMV5 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0


# # train Base (RFDB)
# CUDA_VISIBLE_DEVICES=2,3 python -u main.py --n_GPUs 2 --n_threads 8 \
# --model CFGN --scale 2 --n_feats 48 --n_resgroups 6 --act identity --block_type base \
# --save CFGN-base_BIX2_F48R6 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# # train Base+ACT
# CUDA_VISIBLE_DEVICES=6,7 python -u main.py --n_GPUs 2 --n_threads 8 \
# --model CFGN --scale 2 --n_feats 48 --n_resgroups 6 --act lrelu --block_type base \
# --save CFGN-base+ACT_BIX2_F48R6 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# # train Base+CFGM
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u main.py --n_GPUs 6 \
# --model CFGN --scale 2 --n_feats 64 --n_resgroups 8 --act identity --block_type CFGM \
# --save CFGN-base+CFGM_BIX2_F64R8 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# # train Base+ACT+CFGM (CFGG)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u main.py --n_GPUs 6 \
# --model CFGN --scale 2 --n_feats 64 --n_resgroups 8 --act lrelu --block_type CFGM \
# --save CFGN-base+ACT+CFGM_BIX2_F64R8 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0



# train SRB (Base, RFDB-L)
CUDA_VISIBLE_DEVICES=1 python -u main.py --n_GPUs 1 --n_threads 8 \
--model CFGN --scale 2 --n_feats 52 --n_resgroups 6 --act identity --block_type base \
--save CFGN_SRB_BIX2_F52R6 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# train CFGM_v1
CUDA_VISIBLE_DEVICES=0,1,2 python -u main.py --n_GPUs 3 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act identity --block_type CFGM_v1 \
--save CFGN_CFGM_v1_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# train CFGM_v2
CUDA_VISIBLE_DEVICES=4,5 python -u main.py --n_GPUs 2 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act identity --block_type CFGM_v2 \
--save CFGN_CFGM_v2_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# train ButterflyConv_v1
CUDA_VISIBLE_DEVICES=0,1,2 python -u main.py --n_GPUs 3 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act identity --block_type ButterflyConv_v1 \
--save CFGN_ButterflyConv_v1_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# train ButterflyConv_v2
CUDA_VISIBLE_DEVICES=0,1,2 python -u main.py --n_GPUs 3 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act identity --block_type ButterflyConv_v2 \
--save CFGN_ButterflyConv_v2_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0





# train SRB + ACT
CUDA_VISIBLE_DEVICES=1 python -u main.py --n_GPUs 1 --n_threads 8 \
--model CFGN --scale 2 --n_feats 52 --n_resgroups 6 --act lrelu --block_type base \
--save CFGN_SRB+ACT_BIX2_F52R6 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# train CFGM_v1 + ACT
CUDA_VISIBLE_DEVICES=2,3 python -u main.py --n_GPUs 2 --n_threads 4 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v1 \
--save CFGN_CFGM_v1+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# train CFGM_v2 + ACT
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 \
--save CFGN_CFGM_v2+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# train ButterflyConv_v1 + ACT
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u main.py --n_GPUs 6 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type ButterflyConv_v1 \
--save CFGN_ButterflyConv_v1+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# train ButterflyConv_v2 + ACT
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u main.py --n_GPUs 6 --n_threads 12 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type ButterflyConv_v2 \
--save CFGN_ButterflyConv_v2+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0



CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --pre_train /home/Leeyegy/yamengxi/EDSR-PyTorch/experiment/CFGN_CFGM_v2+ACT_BIX2_F64R9_2021-05-07_22:13:13/model/model_best.pt \
--save CFGN_CFGM_v2+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 --test_only # --test_every 0



# # train Base+ACT
# CUDA_VISIBLE_DEVICES=2 python -u main.py --n_GPUs 1 --n_threads 8 \
# --model CFGN --scale 2 --n_feats 52 --n_resgroups 6 --act lrelu --block_type base \
# --save CFGN-base+ACT_BIX2_F52R6 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# # train Base+CFGM
# CUDA_VISIBLE_DEVICES=0,1,2 python -u main.py --n_GPUs 3 \
# --model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act identity --block_type CFGM \
# --save CFGN-base+CFGM_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100 --batch_size 64 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# # train Base+ACT+CFGM (CFGN)
# CUDA_VISIBLE_DEVICES=0,1,2 python -u main.py --n_GPUs 3 \
# --model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM \
# --save CFGN-base+ACT+CFGM_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100 --batch_size 3 --patch_size 128 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0



CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 4 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --pre_train /home/Leeyegy/yamengxi/EDSR-PyTorch/experiment/CFGN_CFGM_v2+ACT_BIX2_F64R9_2021-05-07_22:13:13/model/model_latest.pt \
--save CFGN_CFGM_v2+ACT_BIX4_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 256 --save_results --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0


# train CFGN scale = 2
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 1 \
--save dilation=1_num_G=32_num_TFC=4_CFGN_CFGM_v2+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 128 --save_results --save_models \
--optimizer ADAM --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0


CUDA_VISIBLE_DEVICES=3,4,5,6 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 \
--save CFGN_CFGM_v2+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 128 --save_results --save_models \
--optimizer ADAM --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0


CUDA_VISIBLE_DEVICES=2,3 python -u main.py --n_GPUs 2 --n_threads 4 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 \
--save CFGN_CFGM_v2+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 128 --patch_size 128 --save_results --save_models \
--optimizer ADAM --lr 0.001 --decay 100-200-300-400-500-600 --epochs 0 # --test_every 0

CUDA_VISIBLE_DEVICES=0,1 python -u main.py --n_GPUs 2 --n_threads 4 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train ??? \
--save CFGN_CFGM_v2+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size ??? --patch_size 128 --save_results --save_models \
--optimizer SGD --lr 0.01 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0



# train CFGN scale = 3
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 3 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train ??? \
--save CFGN_CFGM_v2+ACT_BIX3_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 192 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# train CFGN scale = 4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 4 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train ??? \
--save CFGN_CFGM_v2+ACT_BIX4_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 256 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# CFGN + dilation = 1
CUDA_VISIBLE_DEVICES=0,1 python -u main.py --n_GPUs 2 --n_threads 4 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 1 \
--save CFGN_CFGM_v2+ACT_dilation=1_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 128 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# CFGN + dilation = 2
CUDA_VISIBLE_DEVICES=2,3 python -u main.py --n_GPUs 2 --n_threads 4 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 2 \
--save CFGN_CFGM_v2+ACT_dilation=2_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 128 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0

# CFGN + dilation = 4
CUDA_VISIBLE_DEVICES=6,7 python -u main.py --n_GPUs 2 --n_threads 4 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 4 \
--save CFGN_CFGM_v2+ACT_dilation=4_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 128 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0



# 去掉部分激活函数的CFGN
CUDA_VISIBLE_DEVICES=0,1,2 python -u main.py --n_GPUs 3 --n_threads 6 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act identity --block_type CFGM_v2 --dilation 3 \
--save CFGN_CFGM_v2_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 128 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0



# CFGN + SRB
CUDA_VISIBLE_DEVICES=3,6,7 python -u main.py --n_GPUs 3 --n_threads 6 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type srb --dilation 3 \
--save CFGN_SRB+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 128 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0




# test CFGN scale = 2
CUDA_VISIBLE_DEVICES=7 python -u main.py --n_GPUs 1 --n_threads 2 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train /data2/yamengxi/CFGN/CFGN-PyTorch/final_models/CFGN_CFGM_v2+ACT_BIX2_F64R9_2023-01-08_14:35:28_model_1305.pt \
--save final_result_dilation=3_num_G=32_num_TFC=3_CFGN_CFGM_v2+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 192 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 --test_only # --test_every 0

CUDA_VISIBLE_DEVICES=4 python -u main.py --n_GPUs 1 --n_threads 2 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train /data2/yamengxi/CFGN/CFGN-PyTorch/final_models/CFGN_CFGM_v2+ACT_BIX2_F64R9_2023-01-08_14:35:28_model_1305.pt \
--save final_result_dilation=3_num_G=32_num_TFC=3_CFGN+_CFGM_v2+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 192 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 --test_only --self_ensemble # --test_every 0

# test CFGN scale = 3
CUDA_VISIBLE_DEVICES=4 python -u main.py --n_GPUs 1 --n_threads 2 \
--model CFGN --scale 3 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train /data2/yamengxi/CFGN/CFGN-PyTorch/final_models/CFGN_CFGM_v2+ACT_BIX3_F64R9_2023-01-17_10:02:03_model_787.pt \
--save final_result_dilation=3_num_G=32_num_TFC=3_CFGN_CFGM_v2+ACT_BIX3_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 288 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 --test_only # --test_every 0

CUDA_VISIBLE_DEVICES=5 python -u main.py --n_GPUs 1 --n_threads 2 \
--model CFGN --scale 3 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train /data2/yamengxi/CFGN/CFGN-PyTorch/final_models/CFGN_CFGM_v2+ACT_BIX3_F64R9_2023-01-17_10:02:03_model_787.pt \
--save final_result_dilation=3_num_G=32_num_TFC=3_CFGN+_CFGM_v2+ACT_BIX3_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 288 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 --test_only --self_ensemble # --test_every 0

# test CFGN scale = 4
CUDA_VISIBLE_DEVICES=4 python -u main.py --n_GPUs 1 --n_threads 2 \
--model CFGN --scale 4 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train /data2/yamengxi/CFGN/CFGN-PyTorch/final_models/CFGN_CFGM_v2+ACT_BIX4_F64R9_2023-01-23_18:42:21_model_688.pt \
--save final_result_dilation=3_num_G=32_num_TFC=3_CFGN_CFGM_v2+ACT_BIX4_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 384 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 --test_only # --test_every 0

CUDA_VISIBLE_DEVICES=6 python -u main.py --n_GPUs 1 --n_threads 2 \
--model CFGN --scale 4 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train /data2/yamengxi/CFGN/CFGN-PyTorch/final_models/CFGN_CFGM_v2+ACT_BIX4_F64R9_2023-01-23_18:42:21_model_688.pt \
--save final_result_dilation=3_num_G=32_num_TFC=3_CFGN+_CFGM_v2+ACT_BIX4_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 384 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 --test_only --self_ensemble # --test_every 0

