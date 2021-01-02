# AGAT: Attribute-Guided Adversarial Training for Robustness to Natural Perturbations
This repository contains code for the AAAI 2021 paper of the same name.  A preprint is available here: https://arxiv.org/abs/2012.01806 .

## Datasets
AGAT is currently setup for 3 robustness benchmark datasets:
1. CLEVR-Singles ([data]() and [data-generation code](https://github.com/tejasG53/CLEVR-Singles))
2. MNIST-RTS (we use the default MNIST dataloader from torchvision.datasets . No download necessary)
3. CIFAR-10-C
    - this dataset was released by Hendrycks and Dietterich [here][https://zenodo.org/record/2535967#.Xaf8uedKj-Y]
    - the CIFAR-10-C repository is [this][https://github.com/hendrycks/robustness]
    - we have adapted parts of the code for our CIFAR-10-C experiments from [TTT (Sun et al. ICML 2020)][https://arxiv.org/abs/1909.13231] whose code is [here][https://github.com/yueatsprograms/ttt_cifar_release] 


