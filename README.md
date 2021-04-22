# Towards Adversarial Patch Analysis and Certified Defense against Crowd Counting
This work is done when working as the research asistant in Big Data Security and Information Itelligence Lab, supervised by Prof.[Pan Zhou](https://scholar.google.com/citations?user=cTpFPJgAAAAJ&hl=en). And we corperate with the researchers in Duke University ([Dr.Wang](https://wangbinghui.net) and [Dr.Li](https://sites.duke.edu/angli))

* This is the official implementation code of paper submitted to ACM MM 2021.

* We are the first work to evaluate the adversarial robustness of crowd counting both theoretically and empirically.

* Thanks for the insightful discussions with [Weizhe Liu](https://weizheliu.github.io)(the author of Coxtext-Aware-Crowd-Counting, CVPR2019)

* Thanks for the repo of Optical Flow Attack (ICCV2019) [repo](https://github.com/anuragranj/flowattack), this repo is inspired from it. 

## Requirement
1. Install pytorch 1.4.0+
2. Python 3.7+

## Data Setup
follow the MCNN or CSRNet repo's steps to build the dataset [MCNN](https://github.com/svishwa/crowdcount-mcnn) [CSRNet](https://github.com/CommissarMa/CSRNet-pytorch)

Download ShanghaiTech Dataset from
[Dropbox](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0) or [Baidu Disk](https://pan.baidu.com/s/101mNo_Vz21IwDYnYTnLQpw) (code: a2v8)   

## Attacked Models
CSRNet: https://github.com/CommissarMa/CSRNet-pytorch

CAN: https://github.com/CommissarMa/Context-Aware_Crowd_Counting-pytorch

MCNN: https://github.com/svishwa/crowdcount-mcnn

CMTL: https://github.com/svishwa/crowdcount-cascaded-mtl

DA-Net: https://github.com/BigTeacher-777/DA-Net-Crowd-Counting

Thanks for these researchers sharing the codes!


## How to Attack?
Please run the python file csr_attack.py

## How to Retrain the Crowd Counting Models?
python3 MCNN_adv_train.py (adversarial training with the generated patch (pristine version))
python3 MCNN_certify_train.py (certificate training MCNN via randomized ablation)

## Want to Gain the Certificate Retrained Models?
dropbox: https://www.dropbox.com/sh/s9v8ojj7pedz4vr/AAChLahRjJ_-ko6kefsSD47ba?dl=0 

[dropbox](https://www.dropbox.com/sh/s9v8ojj7pedz4vr/AAChLahRjJ_-ko6kefsSD47ba?dl=0)

