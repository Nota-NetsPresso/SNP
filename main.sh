CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\
    python3 -m torch.distributed.launch --nproc_per_node 8 --master_addr="127.0.0.1" --master_port=12345 \
        main.py --model deit_tiny_patch16_224 \
                --batch-size 256 \
                --epochs 3 \
                --output_dir ./output \
                > ./txt_logs/training_test.txt 2>&1 &