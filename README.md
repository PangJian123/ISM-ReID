
# HAZY RE-ID: AN INTERFERENCE SUPPRESSION MODEL FOR DOMAIN ADAPTATION PERSON RE-IDENTIFICATION UNDER INCLEMENT WEATHER CONDITION.

This package contains the source code which is associated with the following paper:

Huafeng Li, Jian Pang, Zhengtao Yu, and Dapeng Tao, “HAZY RE-ID: AN INTERFERENCE SUPPRESSION MODEL FOR DOMAIN ADAPTATION PERSON RE-IDENTIFICATION UNDER INCLEMENT WEATHER CONDITION.” accepted as oral on ICME 2021.

Edited by Jian Pang

Usage of this code is free for research purposes only. 

Thank you.

# Requirements:
    CUDA  10.2
    Python  3.8
    Pytorch  1.6.0
    torchvision  0.2.2
    numpy  1.19.0

# Get Started
## 1.Install:
    download the code
    git clone https://github.com/PangJian123/ISM-ReID.git
    cd ISM-ReID
    
## 2.Datasets and the pre-trained 
- Prepare datasets and the pre-trained model please refer to https://github.com/PangJian123/fast-reid
- Download the synthetic hazy datasets  through the links below:
*Hazy-DukeMTMC-reID*:[Baidu Pan](*) 
*Hazy-Market1501*:[Baidu Pan](*) 

## 3.Run the training file:
        sh pre_train.sh (Supervised training on source domain)
        sh train.sh (Training ISM)

# Contact:
    Don't hesitate to contact me if you meet any problems when using this code.

    Jian Pang
    Faculty of Information Engineering and Automation
    Kunming University of Science and Technology                                                           
    Email: pangjian@stu.kust.edu.cn

# Acknowledgements
Our code is based on https://github.com/michuanhaohao/reid-strong-baseline.
