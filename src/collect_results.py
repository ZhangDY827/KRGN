import os
import os.path as osp

model_name = 'CFGN+'

datasets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']

datasets_root_dir = '/data2/yamengxi/CFGN/datasets_for_EDSR-PyTorch'

os.makedirs('HR', exist_ok=True)
for dataset in datasets:
    HR_dir = osp.join(datasets_root_dir, 'benchmark', dataset, 'HR')
    os.symlink(HR_dir, osp.join('./HR', dataset))


for dataset in datasets:
    os.makedirs(osp.join(f'./SR/BI/{model_name}', dataset), exist_ok=True)

for mid_dir in os.listdir('../experiment'):
    if mid_dir.find('final_result') >= 0 and mid_dir.find('BIX') >= 0 and mid_dir.find(model_name) >= 0:
        scale = mid_dir[mid_dir.find('BIX') + 3]
        for dataset in datasets:
            SR_dir = osp.abspath(osp.join('../experiment', mid_dir, dataset))
            os.symlink(SR_dir, osp.join(f'./SR/BI/{model_name}', dataset, 'x'+scale))
