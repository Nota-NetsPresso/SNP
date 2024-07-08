# [[ECCV 2024](https://eccv.ecva.net/)] [SNP: Structured Neuron-level Pruning to Preserve Attention Scores](https://arxiv.org/abs/2404.11630)

This repository contains the official implementation of the paper "[SNP: Structured Neuron-level Pruning to Preserve Attention Scores](https://arxiv.org/abs/2404.11630)" accepted at the [European Conference on Computer Vision (ECCV) 2024](https://eccv.ecva.net/).

## Introduction
Structured Neuron-level Pruning (SNP) prunes neurons with less informative attention scores and eliminates redundancy among heads. Our proposed method effectively compresses and accelerates Transformer-based models for both edge devices and server processors. SNP with head pruning could compress the DeiT-Base by 80\% of the parameters and computational costs and achieve 3.85× faster inference speed on RTX3090 and 4.93× on Jetson Nano.

<div align="center">
    <img src="./fig/perf_latency_params_jetson_nano.png" alt="Description1" style="width:39%; display: inline-block;">
    <img src="./fig/Fig1.PNG" alt="Description2" style="width:39%; display: inline-block;">
</div>

## Proposed Method

Structured Neuron-level Pruning (SNP) prunes graphically connected query and key layers having the least informative attention scores while preserving the overall attention scores. Value layers, which can be pruned independently, are pruned to eliminate inter-head redundancy.

<div align="center">
    <img src="./fig/proposed methods.PNG" alt="Description1" style="width:65%; display: inline-block;">
    <img src="./fig/fig3_attention_maps.PNG" alt="Description2" style="width:30%; display: inline-block;">
</div>

<!-- <div style="text-align: center;">
    <img src="./fig/tab.PNG" alt="Description" style="width: 90%;height:60%">
</div> -->

## Benchmark

<center>

| Model               | Top-1 (%)         | GFLOPs           | Raspberry Pi 4B (.onnx) | Jetson Nano (.trt)  | Xeon Silver 4210R (.pt) | RTX 3090 (.pt)        |
|---------------------|-------------------|------------------|-------------------------|----------------------|-------------------------|-----------------------|
| **DeiT-Tiny**           | 72.20             | 1.3              | 139.13                  | 41.03                | 34.74                   | 18.65                 |
| **+ SNP (Ours)**    | **70.29**         | **0.6**          | **81.63 (1.70×)**       | **26.67 (1.54×)**    | **25.25 (1.38×)**       | **17.82 (1.05×)**     |
| **DeiT-Small**          | 79.80             | 4.6              | 401.27                  | 99.32                | 53.37                   | 46.13                 |
| **+ SNP (Ours)**    | **78.52**         | **2.0**          | **199.15 (2.01×)**      | **45.51 (2.18×)**    | **38.57 (1.38×)**       | **32.91 (1.40×)**     |
| **+ SNP (Ours)**    | **73.32**         | **1.3**          | **136.68 (2.94×)**      | **32.03 (3.10×)**    | **33.46 (1.60×)**       | **26.98 (1.71×)**     |
| **DeiT-Base**           | 81.80             | 17.6             | 1377.71                 | 293.29               | 122.03                  | 151.35                |
| **+ SNP (Ours)**    | **79.63**         | **6.4**          | **565.68 (2.44×)**      | **132.55 (2.21×)**   | **64.65 (1.89×)**       | **72.96 (2.07×)**     |
| **+ SNP (Ours) + Head** | **79.12**         | **3.5**          | **307.00 (4.48×)**      | **59.47 (4.93×)**    | **46.09 (2.65×)**       | **39.31 (3.85×)**     |
| **EfficientFormer-L1**  | 79.20             | 1.3              | 169.13                  | 30.95                | 43.75                   | 26.19                 |
| **+ SNP (Ours)**    | **75.53**         | **0.6**          | **95.12 (1.78×)**       | **19.78 (1.56×)**    | **38.25 (1.14×)**       | **17.24 (1.52×)**     |
| **+ SNP (Ours)**    | **74.51**         | **0.5**          | **82.60 (2.05×)**       | **17.76 (1.74×)**    | **35.15 (1.24×)**       | **16.01 (1.64×)**     |

</center>

## Try SNP on your own Model
<!-- <div align="center">
    <a href="https://netspresso.ai/?utm_source=git&utm_medium=banner_py&utm_campaign=np_renew" target="_blank"><img src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/banner/NetsPresso2.0_banner.png"/>
</div>
</br> -->

<div align="center">
    <a href="https://github.com/Nota-NetsPresso/PyNetsPresso/tree/develop" target="_blank"><img src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/banner/NetsPresso2.0_banner.png"/>
</div>
</br>


## Citation
```bibtex
@article{shim2024snp,
  title={SNP: Structured Neuron-level Pruning to Preserve Attention Scores},
  author={Shim, Kyunghwan and Yun, Jaewoong and Choi, Shinkook},
  journal={arXiv preprint arXiv:2404.11630},
  year={2024},
  url={https://arxiv.org/abs/2404.11630}
}
```