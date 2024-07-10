# [[ECCV 2024](https://eccv.ecva.net/)] [SNP: Structured Neuron-level Pruning to Preserve Attention Scores](https://arxiv.org/abs/2404.11630)

This repository contains the official implementation of the paper "[SNP: Structured Neuron-level Pruning to Preserve Attention Scores](https://arxiv.org/abs/2404.11630)" accepted at the [European Conference on Computer Vision (ECCV) 2024](https://eccv.ecva.net/).

## Introduction
Structured Neuron-level Pruning (SNP) prunes neurons with less informative attention scores and eliminates redundancy among heads. Our proposed method effectively compresses and accelerates Transformer-based models for both edge devices and server processors. SNP with head pruning could compress the DeiT-Base by 80\% of the parameters and computational costs and achieve 3.85× faster inference speed on RTX3090 and 4.93× on Jetson Nano.

<div align="center">
    <img src="./fig/perf_latency_params_jetson_nano.png" alt="Description1" style="width:39%; display: inline-block;">
    <img src="./fig/Fig1.PNG" alt="Description2" style="width:39%; display: inline-block;">
</div>

## Proposed Method

Structured Neuron-level Pruning (SNP) prunes graphically connected query and key layers having the least informative attention scores while preserving the overall attention scores. Value layers, which can be pruned independently, are pruned to eliminate inter-head redundancy. For more detailed information please refer to the main [paper](https://arxiv.org/abs/2404.11630).

<div align="center">
    <img src="./fig/proposed methods.PNG" alt="Description1" style="width:100%;">
    <!-- <img src="./fig/fig3_attention_maps.PNG" alt="Description2" style="width:30%; display: inline-block;"> -->
</div>

<!-- <div style="text-align: center;">
    <img src="./fig/tab.PNG" alt="Description" style="width: 90%;height:60%">
</div> -->

## Benchmark

<center>

| <sub>Model</sub>               | <sub>Top-1 (%)</sub>         | <sub>GFLOPs</sub>           | <sub>Raspberry Pi 4B (.onnx)</sub> | <sub>Jetson Nano (.trt)</sub>  | <sub>Xeon Silver 4210R (.pt)</sub> | <sub>RTX 3090 (.pt)</sub>         |
|---------------------|-------------------|------------------|-------------------------|----------------------|-------------------------|-----------------------|
| <sub>DeiT-Tiny</sub>           | <sub>72.2</sub>             | <sub>1.3</sub>              | <sub>139.1</sub>                  | <sub>41.0</sub>                | <sub>34.7</sub>                   | <sub>18.7</sub>                 |
| <sub>+ SNP (Ours)</sub>        | <sub>70.2</sub>             | <sub>0.6</sub>              | <sub>81.6 (1.70×)</sub>           | <sub>26.7 (1.54×)</sub>        | <sub>25.3 (1.38×)</sub>           | <sub>17.8 (1.05×)</sub>         |
| <sub>DeiT-Small</sub>          | <sub>79.8</sub>             | <sub>4.6</sub>              | <sub>401.3</sub>                  | <sub>99.3</sub>                | <sub>53.4</sub>                   | <sub>46.1</sub>                 |
| <sub>+ SNP (Ours)</sub>        | <sub>78.5</sub>             | <sub>2.0</sub>              | <sub>199.2 (2.01×)</sub>          | <sub>45.5 (2.18×)</sub>        | <sub>38.6 (1.38×)</sub>           | <sub>32.9 (1.40×)</sub>         |
| <sub>+ SNP (Ours)</sub>        | <sub>73.3</sub>             | <sub>1.3</sub>              | <sub>136.7 (2.94×)</sub>          | <sub>32.0 (3.10×)</sub>        | <sub>33.5 (1.60×)</sub>           | <sub>27.0 (1.71×)</sub>         |
| <sub>DeiT-Base</sub>           | <sub>81.8</sub>             | <sub>17.6</sub>             | <sub>1377.7</sub>                 | <sub>293.3</sub>               | <sub>122.0</sub>                  | <sub>151.4</sub>                |
| <sub>+ SNP (Ours)</sub>        | <sub>79.6</sub>             | <sub>6.4</sub>              | <sub>565.7 (2.44×)</sub>          | <sub>132.6 (2.21×)</sub>       | <sub>64.7 (1.89×)</sub>           | <sub>73.00 (2.07×)</sub>         |
| <sub>+ SNP (Ours) + Head</sub> | <sub>79.1</sub>             | <sub>3.5</sub>              | <sub>307.0 (4.48×)</sub>          | <sub>59.5 (4.93×)</sub>        | <sub>46.1 (2.65×)</sub>           | <sub>39.3 (3.85×)</sub>         |
| <sub>EfficientFormer-L1</sub>  | <sub>79.2</sub>             | <sub>1.3</sub>              | <sub>169.1</sub>                  | <sub>31.0</sub>                | <sub>43.8</sub>                   | <sub>26.2</sub>                 |
| <sub>+ SNP (Ours)</sub>        | <sub>75.5</sub>             | <sub>0.6</sub>              | <sub>95.1 (1.78×)</sub>           | <sub>19.8 (1.56×)</sub>        | <sub>38.3 (1.14×)</sub>           | <sub>17.2 (1.52×)</sub>         |
| <sub>+ SNP (Ours)</sub>        | <sub>74.5</sub>             | <sub>0.5</sub>              | <sub>82.6 (2.05×)</sub>           | <sub>17.8 (1.74×)</sub>        | <sub>35.2 (1.24×)</sub>           | <sub>16.0 (1.64×)</sub>         |

</center>


## Installation
```
git clone https://github.com/Nota-NetsPresso/SNP.git
pip install -r requirements.txt
```

## Getting Started
### Sign Up for  [NetsPresso](https://netspresso.ai/) 

To compress the DeiT model using SNP, you need to sign up for a NetsPresso account. You can sign up [here](https://netspresso.ai/) or go directly to the [Sign Up page](https://account.netspresso.ai/signup).

### Simple Run
Following steps compress the DeiT-T model using SNP and train it for 20 epochs:
1. Run the main script:
    ```
    bash main.sh
    ```
2. When prompted, enter your NetsPresso user information:
    ```
    Please enter your NetsPresso Email:
    Please enter your NetsPresso Password:
    ```
3. Enter the path to your ImageNet dataset:
    ```
    Please enter the path to your ImageNet dataset:
    ```

### Reproduce results on ImageNet
<details>
<summary>Reproducing the results of DeiT-T with 0.6 GFLOPs and 70.29% on ImageNet.</summary>

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\
        python -m torch.distributed.launch --nproc_per_node 8 --master_addr="127.0.0.1" --master_port=12345 \
            train.py --model "./reported_models/compressed_models/DeiT-T.pt" \
                    --lr 0.001 \
                    --batch-size 256 \
                    --epochs 300 \
                    --output_dir ${OUPUT_DIR} \
                    --data-path  ${IMAGENET_PATH}\
                    > ./txt_logs/training_deit_t.txt 2>&1 &

</details>

<details>
<summary> Reproducing the results of DeiT-S with 2 GFLOPs and 78.52% on ImageNet.</summary>

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\
            python -m torch.distributed.launch --nproc_per_node 8 --master_addr="127.0.0.1" --master_port=12345 \
                train.py --model "./reported_models/compressed_models/DeiT-S_2GFLOPs.pt" \
                        --lr 0.001 \
                        --batch-size 256 \
                        --epochs 300 \
                        --output_dir ${OUPUT_DIR} \
                        --data-path  ${IMAGENET_PATH}\
                        > ./txt_logs/training_deit_s_2GFLOPs.txt 2>&1 &
                        
</details>

<details>
<summary>Reproducing the results of DeiT-S with 1.27 GFLOPs and 73.32% on ImageNet.</summary>

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\
            python -m torch.distributed.launch --nproc_per_node 8 --master_addr="127.0.0.1" --master_port=12345 \
                train.py --model "./reported_models/compressed_models/DeiT-S_1_27GFLOPs.pt" \
                        --lr 0.001 \
                        --batch-size 256 \
                        --epochs 300 \
                        --output_dir ${OUPUT_DIR} \
                        --data-path  ${IMAGENET_PATH}\
                        > ./txt_logs/training_deit_s_1_27GFLOPs.txt 2>&1 &
                        
</details>


### Overall Instructions for SNP
1. To compress the DeiT model, use the following command:

    ```bash 
    python compress.py --NetsPresso-Email ${USER_NAME} \
                        --NetsPresso-Pwd ${USER_PWD} \
                        --model deit_tiny_patch16_224 \
                        --data-path ${IMAGENET_PATH}\
                        --output_dir ${OUPUT_DIR} \
                        --num-imgs-snp-calculation 64\
    ```

2. To train the compressed model (saved in the `compressed` directory within `output_dir`), use the following command:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\
        python -m torch.distributed.launch --nproc_per_node 8 --master_addr="127.0.0.1" --master_port=12345 \
            train.py --model "${OUPUT_DIR}/compressed/compressed.pt" \
                    --batch-size 256 \
                    --epochs 300 \
                    --output_dir ${OUPUT_DIR} \
                    --data-path  ${IMAGENET_PATH}\
                    > ./txt_logs/training_test.txt 2>&1 &
    ```

## Try SNP on your own Model

<div align="center">
    <a href="https://netspresso.ai/?utm_source=git&utm_medium=banner_py&utm_campaign=np_renew" target="_blank"><img src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/banner/NetsPresso2.0_banner.png"/>
</div>

<br>

<div align="center">
  ☀️ NetsPresso Model Zoo ☀️ <br>
      <a href="https://github.com/Nota-NetsPresso/ModelZoo-YOLOFastest-for-ARM-U55-M85"> YOLO Fastest </a>
    | <a href="https://github.com/Nota-NetsPresso/yolox_nota"> YOLOX </a>
    | <a href="https://github.com/Nota-NetsPresso/ultralytics_nota"> YOLOv8 </a> 
    | <a href="https://github.com/Nota-NetsPresso/ModelZoo-YOLOv7"> YOLOv7 </a> 
    | <a href="https://github.com/Nota-NetsPresso/yolov5_nota"> YOLOv5 </a> 
    | <a href="https://github.com/Nota-NetsPresso/PIDNet_nota"> PIDNet </a>     
    | <a href="https://github.com/Nota-NetsPresso/pytorch-cifar-models_nota"> PyTorch-CIFAR-Models</a>
</div>
</br>

```Python
from netspresso import NetsPresso
from netspresso.enums import CompressionMethod, GroupPolicy, LayerNorm, Policy
from netspresso.clients.compressor.v2.schemas import Options

# Step 0: Login to NetsPresso
netspresso = NetsPresso(email=args.NetsPresso_Email, password=args.NetsPresso_Pwd)

# Step 1: Declare the compressor
compressor = netspresso.compressor_v2()

# Step 2: Upload the model
# Provide the path to your model and specify the input shape
model = compressor.upload_model(
    input_model_path=${MODEL_PATH},
    input_shapes=[{"batch": 1, "channel": 3, "dimension": [224, 224]}],
)

# Step 3: Select the compression method
# Specify the compression method and options
compression_info = compressor.select_compression_method(
    model_id=model.ai_model_id,
    compression_method=CompressionMethod.PR_SNP,
    options=Options(
        policy=Policy.AVERAGE,
        layer_norm=LayerNorm.TSS_NORM,
        group_policy=GroupPolicy.NONE,
        reshape_channel_axis=-1,
    ),
)

# Step 4: Load the compression ratio for each layer
# Assign the compression ratio for each available layer
for available_layer in compression_info.available_layers:
    available_layer.values = [${COMPRESS_RATIO}[available_layer.name]]

# Step 5: Compress the model
# Perform the compression and save the compressed model
compressed_model_info = compressor.compress_model(
    compression=compression_info,
    output_dir=${SAVE_DIR},
)

# Load the compressed model
compressed_model=torch.load(compressed_model_info.compressed_model_path)
```

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