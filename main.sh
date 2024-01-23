CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py    --model deit_tiny_patch16_224 \
                                                        --batch-size 256 \
                                                        --epochs 3 \
                                                        --output_dir ./output