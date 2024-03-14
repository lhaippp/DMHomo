# DMHomo: Learning Homography with Diffusion Models
[dl.acm.org/doi/10.1145/3652207](https://dl.acm.org/doi/10.1145/3652207)

## To-Do List
  1. Release the pre-trained models and inference code for the **HEM** (Homography Estimator Module) and **DGM** (Data Generator Module).
  2. Release the generated dataset based on the [**CA-Homo Dataset**](https://github.com/JirongZhang/DeepHomography) and the trained **DGM**.
  3. Release the training scripts for **HEM** and **DGM**.

## Thanks
Our framework builds upon previous benchmarking works; we offer our gratitude to them, including, but not limited to:
- [denoising-diffusion-pytorch (lucidrains)](https://github.com/lucidrains/denoising-diffusion-pytorch)
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)
- [HomoGAN](https://github.com/megvii-research/HomoGAN)
- [PDCNet](https://github.com/PruneTruong/DenseMatching)

## Citation
If you use this code or ideas from our paper for your research, please cite our paper:
```
@article{li2024dmhomo,
  title={DMHomo: Learning Homography with Diffusion Models},
  author={Li, Haipeng and Jiang, Hai and Luo, Ao and Tan, Ping and Fan, Haoqiang and Zeng, Bing and Liu, Shuaicheng},
  journal={ACM Transactions on Graphics},
  year={2024},
  publisher={ACM New York, NY}
}
```
