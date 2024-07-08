# [[ECCV 2024](https://eccv.ecva.net/)] [SNP: Structured Neuron-level Pruning to Preserve Attention Scores](https://arxiv.org/abs/2404.11630)
This repository contains the official implementation of the paper "[SNP: Structured Neuron-level Pruning to Preserve Attention Scores](https://arxiv.org/abs/2404.11630)" accepted at the [European Conference on Computer Vision (ECCV) 2024](https://eccv.ecva.net/).

## Introduction

<div style="float: right; width: 40%;">
    <img src="./fig/Fig1.PNG" alt="Description" style="width: 100%;">
</div>

Structured Neuron-level Pruning (SNP) prunes graphically connected query and key layers having the least informative attention scores while preserving the overall attention scores. Value layers, which can be pruned independently, are pruned to eliminate inter-head redundancy. Our proposed method effectively compresses and accelerates Transformer-based models for both edge devices and server processors. SNP with head pruning could compress the DeiT-Base by 80\% of the parameters and computational costs and achieve 3.85× faster inference speed on RTX3090 and 4.93× on Jetson Nano.

<div style="text-align: center;">
    <img src="./fig/proposed methods.PNG" alt="Description1" style="width:65%; display: inline-block;">
    <img src="./fig/fig3_attention_maps.PNG" alt="Description2" style="width:30%; display: inline-block;">
</div>

<!-- <div style="text-align: center;">
    <img src="./fig/tab.PNG" alt="Description" style="width: 90%;height:60%">
</div> -->



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