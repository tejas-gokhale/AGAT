# AGAT: Attribute-Guided Adversarial Training for Robustness to Natural Perturbations
This repository contains code for the AAAI 2021 paper of the same name.  A preprint is available here: https://arxiv.org/abs/2012.01806 .

The CLEVR-Singles dataset released as part of our publication can be found in a separate repository [https://github.com/tejas-gokhale/CLEVR-Singles][https://github.com/tejas-gokhale/CLEVR-Singles]

## Datasets
This repository is currently setup for the CIFAR-10-C benchmark (Hendrycks and Dietterich) 
- CIFAR-10-C dataset [download][https://zenodo.org/record/2535967#.Xaf8uedKj-Y]
- CIFAR-10-C codebase (not required) [this][https://github.com/hendrycks/robustness]

We have adapted parts of the code for our CIFAR-10-C experiments from [TTT (Sun et al. ICML 2020)][https://arxiv.org/abs/1909.13231] whose code is [here][https://github.com/yueatsprograms/ttt_cifar_release]

## Pre-Requisites
pip install the following:
```
pytorch (works with 1.6.0), numpy, tqdm, skimage , colorama
```


## Inference
Pretrained models are found in `./results/`.
Please run:
`python test_ours.py --shared layer2 --rotation_type expand --group_norm 8 \
--ckpt <ckpt_path>`

For instance, to reproduce the "blur+noise" model from the paper (Table), replace `<ckpt_path>` by `./results/cifar10_agat_blur_noise/ckpt_final.pth`

## Training
Coming soon!

## Reference
If you find AGAT or the CLEVR-Singles dataset useful, please use the following citation:
```
@article{gokhale2020attribute,
  title={Attribute-Guided Adversarial Training for Robustness to Natural Perturbations},
  author={Gokhale, Tejas and Anirudh, Rushil and Kailkhura, Bhavya and Thiagarajan, Jayaraman J and Baral, Chitta and Yang, Yezhou},
  journal={arXiv preprint arXiv:2012.01806},
  year={2020}
}
```

