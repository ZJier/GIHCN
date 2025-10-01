# GIHCN

**Generative Information-guided Heterogeneous Cross-fusion Network with Contrastive Learning for Multimodal Remote Sensing Image Classification**

This work has been accepted by IEEE Transactions on Circuits and Systems for Video Technology (IEEE TCSVT: https://ieeexplore.ieee.org/document/11177538).

## Config
Ubuntu 22.04.4 LTS (GNU/Linux 6.6.87.2-microsoft-standard-WSL2 x86_64)

Python 3.10.13-----Torch 2.1.1

You must install the relevant libraries for Mamba.

## Other
For convenience, we have written all modules related to the proposed method into the GIHCN file.

We only open-source the proposed method and the loss function. This is because we encourage authors to use the same set of training and testing codes to conduct relevant experiments on different algorithms. For example, you can easily integrate our work into the code framework you are currently using, rather than having one code framework for each comparative algorithm.

We do not mind if the algorithm's classification performance in new environments/versions/frameworks is lower or higher than that reported in the paper.

For the generative information of different datasets, please refer to Baidu Netdisk: https://pan.baidu.com/s/1Vbo09Hx38W8RV6bDesP6Sg?pwd=RSGI; Code 1: https://github.com/chenning0115/spectraldiff_diffusion/ or Code 2: https://github.com/ZJier/DKDMN.

Although we have provided all the generative information of four datasets, we hope that you can run the code related to the diffusion model to facilitate your use of more datasets.

Since the generative information of different modalities is extracted separately, Code 1 and Code 2 can realize the generative information extraction for different modalities.

## Citation
If this work is helpful to you, you can refer to the following.

Zhang J, Zhao F, Liu H, et al. Generative information-guided heterogeneous cross-fusion network with contrastive learning for multimodal remote sensing image classification[J]. IEEE Transactions on Circuits and Systems for Video Technology, 2025, Early Access. DOI: https://doi.org/10.1109/TCSVT.2025.3614153.

## Previous work: 
If you need to learn more about the diffusion model methods of HSIC, you can refer to our previous work (DKDMN) and SpectralDiff.

Zhang J., Zhao F., Liu H., et al. Data and knowledge-driven deep multiview fusion network based on diffusion model for hyperspectral image classification[J]. Expert Systems with Applications, 2024, 249: 123796. DOI: https://doi.org/10.1016/j.eswa.2024.123796.

## Special Thanks!
Thanks to the SpectralDiff authors for their contributions to the hyperspectral image classification community, and welcome to cite their latest work!

Chen N, Yue J, Fang L, et al. SpectralDiff: A generative framework for hyperspectral image classification with diffusion models[J]. IEEE Transactions on Geoscience and Remote Sensing, 2023, 61: 5522416. DOI: https://doi.org/10.1109/TGRS.2023.3310023.
