# KRGN
PyTorch implementation of "Towards lightweight image super-resolution via re-parameterized kernel recalibration"

## Introduction
Over the past decade, the exploration of lightweight image super-resolution (SR) networkshas garnered increasing attention, propelled by a notable demand to deploy models on devices with constrained computing resources. A prevalent technique for reducing parameters in SR networks is depth-wise convolution. However, such methods compromise the model's expressive power as it operates on individual segmented features. To address this issue, we propose a novel kernel recalibration strategy that relates the kernel of one channel to the parameters of other channels, allowing the kernels to redirect their focus towards previously unattended sections of the model, instead of the isolated input channel only. Based on this technique, a novel progressive multi-scale recalibration block (PMRB) is proposed to capture more discriminative features with various multi-scale receptive fields. After training, the parameters brought by the kernel recalibration can be re-parameterized to align with the original convolution. Consequently, the resulting block maintains the same inferencecosts as the original block, yet offers improved performance. Additionally, we introducea lightweight SR network, referred to as the kernel recalibration-guided network (KRGN), with the primary objective of attaining a superior equilibrium between effectiveness and availability. Comprehensive experiments validate the competitive results of our proposed KRGN while employing fewer parameters, even achieving only one-third of the computa-tional requirements of some state-of-the-art lightweight methods.

![KRGN](/imgs/model.png)
The architecture of our proposed kernel recalibration-guided network (KRGN).

## Usage
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [RCAN(PyTorch)](https://github.com/yulunzhang/RCAN/tree/master).

### Train (Demo)

Train KRGN with upsampling scale = 4
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --n_GPUs 4 --n_threads 8 \
--model KRGN --scale 4 --n_feats 64 --n_resgroups 9 --act lrelu --dilation 3 --pre_train pre_train_model_path \
--save train_ckpt --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 256 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0
```

### Inference (Demo)
If you want to train KRGN, please ``cd */KRGN/src`` and run the following command in bash.

Test KRGN with upsampling scale = 4
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --n_GPUs 4 --n_threads 8 \
--model KRGN --scale 4 --n_feats 64 --n_resgroups 9 --act lrelu --dilation 3 --pre_train pre_train_model_path \
--save train_ckpt --data_test Set5+Set14+B100+Urban100+Manga109 --save_results --save_models
```

## Results
### Quantitative Results
![Visual_PSNR_SSIM_BI](/imgs/result2.png)

### Quantitative Results
Running the file ```kl.py ``` to generate the visualization of the similarity matrix of Figure 4 in the paper.
![KL](/imgs/kl.png)
