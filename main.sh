#!/bin/bash

read -p "Please enter your NetsPresso Email: " USER_NAME
read -s -p "Please enter your NetsPresso Password: " USER_PWD
echo
read -p "Please enter the path to your ImageNet dataset: " IMAGENET_PATH
echo

export USER_NAME
export USER_PWD
export IMAGENET_PATH

export OUPUT_DIR="./output"

python3 compress.py --NetsPresso-Email ${USER_NAME} \
                    --NetsPresso-Pwd ${USER_PWD} \
                    --model deit_tiny_patch16_224 \
                    --data-path ${IMAGENET_PATH}\
                    --output_dir ${OUPUT_DIR} \
                    --num-imgs-snp-calculation 64\

# Check if compress.py ran successfully
if [ $? -ne 0 ]; then
  echo "compress.py failed to execute. Exiting."
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\
    python3 -m torch.distributed.launch --nproc_per_node 8 --master_addr="127.0.0.1" --master_port=12345 \
        train.py --model "${OUPUT_DIR}/compressed/compressed.pt" \
                 --batch-size 256 \
                 --epochs 20 \
                 --output_dir ${OUPUT_DIR} \
                 --data-path  ${IMAGENET_PATH}\
                 > ./txt_logs/training_test.txt 2>&1 &

# Check if train.py ran successfully
if [ $? -ne 0 ]; then
  echo "train.py failed to execute. Exiting."
  exit 1
fi

echo "Both scripts ran successfully."